% ECE 592 Project: Differential-Drive Navigation with Hybrid A* + MPC iLQR

% Features:
%   - Occupancy-grid map with obstacles
%   - Hybrid A* planner (8-connected + heading bins)
%   - Receding-horizon (MPC-style) iLQR controller
%   - Automatic replanning when deviation from path is large
%   - PID baseline controller
%   - True plant with model mismatch + actuator lag + heading bias
%   - Additive state noise
%   - Obstacle-aware iLQR cost using distance map
%   - Control-smoothing penalty in iLQR
%   - RMSE & control-effort metrics (MPC iLQR vs PID)

function ilqr_diff_drive_upgrade()
    clc; clear; close all; rng(0);


    %% PARAMETERS
    dt          = 0.05;          % time step [s]
    horizon_mpc = 2.0;           % iLQR horizon [s]
    Nh          = round(horizon_mpc/dt);   % steps per MPC horizon

    total_sim   = 8.0;           % total simulation time [s]
    total_steps = round(total_sim/dt);

    vmax    = 0.8;               % max linear velocity [m/s]
    wmax    = 2.0;               % max angular velocity [rad/s]
    robot_r = 0.15;              % robot radius [m]

    % State cost (tracking deviation from nominal)
    Q  = diag([10, 10, 1]);
    Qf = diag([200, 200, 10]);

    % Control cost (effort)
    R  = diag([0.05, 0.05]);

    % Control smoothing (rate) penalty du' * Rd * du
    Rd = 0.01 * eye(2);

    % Obstacle cost parameters
    w_obs    = 10;       % obstacle cost weight
    sigma_obs = 0.3;     % decay length [m]

    % Map parameters
    map_res  = 0.05;
    xlim_map = [-3, 3];
    ylim_map = [-2.5, 2.5];
    Hbins    = 16;           % heading bins for Hybrid A*

    % Noise / model mismatch
    noise_on  = true;
    noise_std = [0.01; 0.01; 0.02];   % [x y theta] noise std dev
    mismatch.v_scale  = 1.10;         % true robot moves 10% faster than model
    mismatch.dt_scale = 1.00;
    mismatch.w_bias   = 0.03;         % constant heading bias [rad/s]
    mismatch.tau_v    = 0.3;          % actuator lag time const for v [s]
    mismatch.tau_w    = 0.2;          % actuator lag time const for w [s]

    % Replanning threshold (distance in meters)
    replanning_threshold = 0.4;

    %% MAP AND DISTANCE MAP
    occ = build_map(xlim_map, ylim_map, map_res);

    % Inflatating occupancy for robot radius (used for planning / collision)
    occ_inf = inflate_occupancy(occ, robot_r, map_res);

    % Distance map for obstacle-aware cost (distance to nearest inflated obstacle)
    dist_map = compute_distance_map(occ_inf, map_res);

    %% START / GOAL
    start = [-2.5; -1.8; 0];
    goal  = [ 2.5;  1.8; 0];

    %% INITIAL PLANNING
    fprintf('Running initial Hybrid A* (8-connected)...\n');
    [path_xy, path_theta] = hybrid_a_star_8conn( ...
        occ_inf, map_res, xlim_map, ylim_map, Hbins, start, goal);

    if isempty(path_xy)
        error('Hybrid A* failed to find an initial path.');
    end
    fprintf('Initial planner found %d waypoints.\n', size(path_xy,1));

    %% NOMINAL TRAJECTORY
    % Making nominal long enough to cover whole simulation + horizon margin
    M_nom         = total_steps + Nh + 1;
    nominal_full  = path_to_nominal_dense(path_xy, path_theta, start, goal, M_nom);
    nominal_sim   = nominal_full(:,1:total_steps+1);  % used for metrics

    %% SHARED NOISE SEQUENCE
    if noise_on
        noise_seq = noise_std(:) .* randn(3, total_steps);
    else
        noise_seq = zeros(3, total_steps);
    end

    %% MPC LOOP WITH iLQR
    u_warm = zeros(2, Nh-1);
    u_warm(1,:) = 0.4;   % initial guess: mild forward motion

    X_mpc    = zeros(3, total_steps+1);
    U_mpc    = zeros(2, total_steps);
    cost_mpc = zeros(1, total_steps);
    X_mpc(:,1) = start;
    xcur = start;

    % Actuator states for MPC plant
    act_mpc.v = 0;
    act_mpc.w = 0;

    fprintf('Running MPC iLQR loop with model mismatch + noise + replanning...\n');
    for k = 1:total_steps

        % If deviation from nominal is large, replanning path (MPC-style)
        idx_closest = find_closest_index(nominal_full, xcur);
        dev = norm(xcur(1:2) - nominal_full(1:2, idx_closest));

        if dev > replanning_threshold
            fprintf('Replanning at step %d (deviation %.2f m)...\n', k, dev);
            [path_xy, path_theta] = hybrid_a_star_8conn( ...
                occ_inf, map_res, xlim_map, ylim_map, Hbins, xcur, goal);

            if isempty(path_xy)
                warning('Replanning failed at step %d, keeping old nominal.', k);
            else
                nominal_full = path_to_nominal_dense(path_xy, path_theta, xcur, goal, M_nom);
                nominal_sim  = nominal_full(:,1:total_steps+1);
            end
        end

        % Extracting local nominal segment around current position
        idx_nom = find_closest_index(nominal_full, xcur);
        nominal_seg = extract_nominal_segment(nominal_full, idx_nom, Nh);

        % Single-horizon iLQR (using nominal model, no noise)
        [Xopt, Uopt, cost_hist] = ilqr_single_horizon( ...
            xcur, u_warm, nominal_seg, Q, R, Qf, Rd, ...
            dt, vmax, wmax, dist_map, xlim_map, ylim_map, map_res, w_obs, sigma_obs);

        cost_mpc(k) = cost_hist(end);

        % Applying first control to TRUE plant (mismatch + actuator lag + noise)
        u_apply      = Uopt(:,1);
        U_mpc(:,k)   = u_apply;
        [x_nom, act_mpc] = dynamics_plant(xcur, u_apply, dt, mismatch, act_mpc);
        xnext        = x_nom + noise_seq(:,k);         % additive state noise
        xnext(3)     = wrapToPi(xnext(3));

        X_mpc(:,k+1) = xnext;
        xcur         = xnext;

        % Warm-start for next iteration: shift horizon and blend
        u_shift        = [Uopt(:,2:end), Uopt(:,end)];
        u_warm         = 0.7 * u_shift + 0.3 * u_warm;

        if mod(k,20) == 0
            fprintf('  step %3d / %3d, latest iLQR cost: %.3f\n', ...
                k, total_steps, cost_hist(end));
        end
    end

    %% PID BASELINE (SAME NOISE/MISMATCH)
    fprintf('Running PID baseline with same noise/mismatch...\n');
    [X_pid, U_pid] = baseline_pid_with_noise( ...
        start, nominal_sim, dt, vmax, wmax, mismatch, noise_seq);

    %% METRICS: RMSE & CONTROL EFFORT
    pos_err_mpc = vecnorm(X_mpc(1:2,:) - nominal_sim(1:2,:), 2, 1);
    pos_err_pid = vecnorm(X_pid(1:2,:)  - nominal_sim(1:2,:), 2, 1);

    rmse_mpc = sqrt(mean(pos_err_mpc.^2));
    rmse_pid = sqrt(mean(pos_err_pid.^2));

    eff_mpc = sum(vecnorm(U_mpc,2,1).^2);
    eff_pid = sum(vecnorm(U_pid,2,1).^2);

    fprintf('\n     Noise / Mismatch Experiment Results     \n');
    fprintf('RMSE position (MPC iLQR): %.3f m\n', rmse_mpc);
    fprintf('RMSE position (PID     ): %.3f m\n', rmse_pid);
    fprintf('Control effort Σ||u||^2 (MPC iLQR): %.3f\n', eff_mpc);
    fprintf('Control effort Σ||u||^2 (PID     ): %.3f\n', eff_pid);
    fprintf('==========================================\n\n');

    %% VISUALIZATION & FIGURES
    fprintf('Animating and creating final figures...\n');
    animate_and_finalfig(occ, occ_inf, map_res, xlim_map, ylim_map, ...
        nominal_sim, X_mpc, U_mpc, X_pid, U_pid, ...
        start, goal, dt, robot_r, cost_mpc, pos_err_mpc, pos_err_pid);

    fprintf('Done.\n');
end

%% MAP GENERATION

function occ = build_map(xlim, ylim, res)
    nx = round((xlim(2)-xlim(1))/res);
    ny = round((ylim(2)-ylim(1))/res);
    occ = zeros(ny, nx);

    % Rectangular obstacles [cx cy wx wy]
    rects = [
        -1.2 -1.5 0.6 2.0;
         0.5 -1.0 0.7 1.8;
        -2.0  0.6 0.6 1.0;
         1.1  0.5 0.8 0.9
    ];
    for i = 1:size(rects,1)
        occ = fill_rect(occ, xlim, ylim, res, rects(i,:));
    end

    % Circular obstacles [cx cy r]
    circs = [
        0.4 -0.6 0.25;
       -1.6  0.9 0.30;
        2.0 -0.4 0.25
    ];
    for i = 1:size(circs,1)
        occ = fill_circle(occ, xlim, ylim, res, circs(i,1), circs(i,2), circs(i,3));
    end
end

function occ = fill_rect(occ, xlim, ylim, res, r)
    cx = r(1); cy = r(2); wx = r(3); wy = r(4);
    nx = size(occ,2); ny = size(occ,1);
    xs = linspace(xlim(1)+res/2, xlim(2)-res/2, nx);
    ys = linspace(ylim(1)+res/2, ylim(2)-res/2, ny);
    for ix = 1:nx
        for iy = 1:ny
            if abs(xs(ix)-cx) <= wx/2 && abs(ys(iy)-cy) <= wy/2
                occ(iy,ix) = 1;
            end
        end
    end
end

function occ = fill_circle(occ, xlim, ylim, res, cx, cy, rr)
    nx = size(occ,2); ny = size(occ,1);
    xs = linspace(xlim(1)+res/2, xlim(2)-res/2, nx);
    ys = linspace(ylim(1)+res/2, ylim(2)-res/2, ny);
    for ix = 1:nx
        for iy = 1:ny
            if hypot(xs(ix)-cx, ys(iy)-cy) <= rr
                occ(iy,ix) = 1;
            end
        end
    end
end

function occ_inf = inflate_occupancy(occ, robot_r, res)
    inflate_cells = ceil(robot_r / res);
    kernel = ones(2*inflate_cells+1);
    occ_inf = conv2(double(occ), kernel, 'same') > 0;
end

function dist_map = compute_distance_map(occ_inf, res)
% Distance (in meters) from each free cell to nearest inflated obstacle.
    if exist('bwdist','file') == 2
        dist_pix = bwdist(occ_inf);      % distance in pixel units to nearest 1
        dist_map = dist_pix * res;       % converts to meters
        dist_map(occ_inf) = 0;           % inside obstacles: distance = 0
    else
        % Fallback: naive O(N^2) distance computation (small maps only)
        [ny, nx] = size(occ_inf);
        [obs_y, obs_x] = find(occ_inf);
        dist_map = inf(ny, nx);
        for iy = 1:ny
            for ix = 1:nx
                if occ_inf(iy,ix)
                    dist_map(iy,ix) = 0;
                else
                    if isempty(obs_x)
                        dist_map(iy,ix) = inf;
                    else
                        d_pix = min(hypot(obs_x - ix, obs_y - iy));
                        dist_map(iy,ix) = d_pix * res;
                    end
                end
            end
        end
    end
end

%% HYBRID A* (8-CONNECTED)

function [path_xy, path_theta] = hybrid_a_star_8conn( ...
        occ_inf, res, xlim, ylim, H, start, goal)

    nx = size(occ_inf,2); ny = size(occ_inf,1);
    xcoords = linspace(xlim(1)+res/2, xlim(2)-res/2, nx);
    ycoords = linspace(ylim(1)+res/2, ylim(2)-res/2, ny);

    % start/goal indices
    [~, sx] = min(abs(xcoords - start(1)));
    [~, sy] = min(abs(ycoords - start(2)));
    [~, gx] = min(abs(xcoords - goal(1)));
    [~, gy] = min(abs(ycoords - goal(2)));
    sx = clamp(sx,1,nx); sy = clamp(sy,1,ny);
    gx = clamp(gx,1,nx); gy = clamp(gy,1,ny);

    start_h = heading_to_bin(start(3), H);
    goal_h  = heading_to_bin(goal(3),  H);

    Nnodes    = nx*ny*H;
    INF       = 1e9;
    gscore    = INF*ones(Nnodes,1);
    fscore    = INF*ones(Nnodes,1);
    came_from = zeros(Nnodes,1);
    open_set  = false(Nnodes,1);

    start_idx = encode_idx(sx, sy, start_h, nx, ny, H);
    goal_idx  = encode_idx(gx, gy, goal_h,  nx, ny, H);

    gscore(start_idx) = 0;
    fscore(start_idx) = heuristic_grid(sx,sy,gx,gy);
    open_set(start_idx) = true;

    % 8-connected moves
    moves = [ 1  0;
              1  1;
              0  1;
             -1  1;
             -1  0;
             -1 -1;
              0 -1;
              1 -1 ];
    move_cost = vecnorm(moves,2,2);

    while any(open_set)
        open_idx = find(open_set);
        [~,pos]  = min(fscore(open_idx));
        current  = open_idx(pos);
        open_set(current) = false;

        [cx,cy,ch] = decode_idx(current,nx,ny,H);
        if current == goal_idx
            path_lin = reconstruct_came(came_from, start_idx, goal_idx);
            [path_xy, path_theta] = decode_path(path_lin, nx, ny, H, xcoords, ycoords);
            return;
        end

        for m = 1:size(moves,1)
            nx_bin = cx + moves(m,1);
            ny_bin = cy + moves(m,2);
            if nx_bin<1 || nx_bin>nx || ny_bin<1 || ny_bin>ny
                continue;
            end
            if occ_inf(ny_bin, nx_bin)
                continue;
            end

            if moves(m,1)==0 && moves(m,2)==0
                nheading = ch;
            else
                theta_move = atan2(moves(m,2), moves(m,1));
                nheading   = heading_to_bin(theta_move, H);
            end
            neighbor = encode_idx(nx_bin,ny_bin,nheading,nx,ny,H);
            tentative = gscore(current) + move_cost(m);
            if tentative < gscore(neighbor)
                gscore(neighbor)    = tentative;
                fscore(neighbor)    = tentative + heuristic_grid(nx_bin,ny_bin,gx,gy);
                came_from(neighbor) = current;
                open_set(neighbor)  = true;
            end
        end
    end

    path_xy    = [];
    path_theta = [];
end

function v = clamp(x,a,b), v = max(a,min(b,x)); end

function idx = encode_idx(xbin,ybin,hbin,nx,ny,H)
    idx = (hbin-1)*(nx*ny) + (ybin-1)*nx + xbin;
end

function [xbin,ybin,hbin] = decode_idx(idx,nx,ny,H)
    idx0 = idx-1;
    xbin = mod(idx0,nx)+1;
    ybin = mod(floor(idx0/nx),ny)+1;
    hbin = floor(idx0/(nx*ny))+1;
end

function d = heuristic_grid(x1,y1,x2,y2)
    d = hypot(x1-x2,y1-y2);
end

function bin = heading_to_bin(theta,H)
    theta = wrapToPi(theta);
    frac  = (theta + pi)/(2*pi);
    bin   = floor(frac*H)+1;
    if bin>H, bin=H; end
    if bin<1, bin=1; end
end

function path_lin = reconstruct_came(came,start_idx,goal_idx)
    path_lin = goal_idx;
    cur = goal_idx;
    while cur ~= start_idx
        cur = came(cur);
        if cur == 0
            path_lin = [];
            return;
        end
        path_lin = [cur; path_lin];
    end
end

function [xy,theta] = decode_path(path_lin,nx,ny,H,xcoords,ycoords)
    L  = numel(path_lin);
    xy = zeros(L,2);
    theta = zeros(L,1);
    for i = 1:L
        [xb,yb,hb] = decode_idx(path_lin(i),nx,ny,H);
        xy(i,1) = xcoords(xb);
        xy(i,2) = ycoords(yb);
        theta(i) = (hb-1)*(2*pi/H)-pi;
    end
end

%% NOMINAL TRAJECTORY HELPERS

function nominal = path_to_nominal_dense(path_xy, path_theta, start, goal, N)
    if isempty(path_xy)
        nominal = repmat(start,1,N);
        return;
    end
    way    = [start(1:2)'; path_xy; goal(1:2)'];
    thetas = [start(3); path_theta(:); goal(3)];
    t  = linspace(0,1,size(way,1));
    tq = linspace(0,1,N);
    xs = interp1(t, way(:,1), tq, 'linear');
    ys = interp1(t, way(:,2), tq, 'linear');
    th = interp1(t, thetas, tq, 'linear');
    nominal = [xs; ys; th];
end

function seg = extract_nominal_segment(nominal_full, idx_center, Nh)
    L = size(nominal_full,2);
    s = idx_center;
    e = s + Nh - 1;
    if e <= L
        seg = nominal_full(:,s:e);
    else
        seg = nominal_full(:,s:L);
        seg = [seg, repmat(nominal_full(:,end),1,e-L)];
    end
end

function idx = find_closest_index(nominal_full, xcur)
    diffs = nominal_full(1:2,:) - xcur(1:2);
    [~, idx] = min(vecnorm(diffs,2,1));
end

%% SINGLE-HORIZON iLQR (USED IN MPC LOOP)

function [Xopt,Uopt,cost_hist] = ilqr_single_horizon( ...
        x0, u_init, nominal, Q,R,Qf,Rd, ...
        dt, vmax,wmax, dist_map, xlim,ylim,res, w_obs, sigma_obs)

    U = u_init;
    X = simulate_traj(x0,U,dt);

    max_iters  = 40;
    alpha_list = [1.0, 0.5, 0.25, 0.1];
    reg        = 1e-6;
    cost_hist  = zeros(max_iters,1);

    for iter = 1:max_iters
        [A,B] = linearize_traj(X,U,dt);
        [lx,lu,lxx,luu,lux] = cost_derivatives( ...
            X,U,nominal,Q,R,Qf,Rd, ...
            dist_map,xlim,ylim,res,w_obs,sigma_obs);

        [K,k,ok,reg] = backward_pass(A,B,lx,lu,lxx,luu,lux,reg);
        if ~ok
            reg = reg * 10;
            continue;
        end

        [Xc,Uc,Jc,success] = forward_search( ...
            x0,X,U,K,k,alpha_list,dt,vmax,wmax, ...
            nominal,Q,R,Qf,Rd, ...
            dist_map,xlim,ylim,res,w_obs,sigma_obs);

        if ~success
            reg = reg * 10;
            continue;
        end

        X = Xc; U = Uc; cost_hist(iter) = Jc;

        if iter > 3 && abs(cost_hist(iter)-cost_hist(iter-1)) < 1e-5
            cost_hist = cost_hist(1:iter);
            break;
        end
    end
    cost_hist = cost_hist(cost_hist~=0);
    Xopt = X;
    Uopt = U;
end

function X = simulate_traj(x0,U,dt)
    N = size(U,2)+1;
    X = zeros(3,N); X(:,1)=x0;
    for t = 1:N-1
        X(:,t+1) = dynamics_nominal(X(:,t),U(:,t),dt);
    end
end

function [Aall,Ball] = linearize_traj(X,U,dt)
    N  = size(X,2); nx=3; nu=2;
    Aall = zeros(nx,nx,N-1);
    Ball = zeros(nx,nu,N-1);
    for t = 1:N-1
        [Aall(:,:,t), Ball(:,:,t)] = linearize_dynamics(X(:,t),U(:,t),dt);
    end
end

function [lx,lu,lxx,luu,lux] = cost_derivatives( ...
        X,U,nominal,Q,R,Qf,Rd, ...
        dist_map,xlim,ylim,res,w_obs,sigma_obs)

    N  = size(X,2); nx=3; nu=2;
    lx = zeros(nx,N);      lu = zeros(nu,N-1);
    lxx = zeros(nx,nx,N);  luu = zeros(nu,nu,N-1);
    lux = zeros(nu,nx,N-1);

    % Stage costs
    for t = 1:N-1
        dx = X(:,t) - nominal(:,t);
        dx(3) = wrapToPi(dx(3));

        % Quadratic state term
        lx(:,t)      = 2*Q*dx;
        lxx(:,:,t)   = 2*Q;

        % Obstacle cost term (approx first-order gradient)
        [c_obs, grad_obs] = obstacle_cost_grad(X(:,t), ...
            dist_map,xlim,ylim,res,w_obs,sigma_obs);
        lx(:,t)      = lx(:,t) + grad_obs;
        % For simplicity second derivative of obstacle was not added (kept 0).

        % Control effort term
        lu(:,t)      = 2*R*U(:,t);
        luu(:,:,t)   = 2*R;

        % Control smoothing (rate) penalty: approx on current control
        if t == 1
            u_prev = [0;0];
        else
            u_prev = U(:,t-1);
        end
        du = U(:,t) - u_prev;
        lu(:,t)    = lu(:,t) + 2*Rd*du;
        luu(:,:,t) = luu(:,:,t) + 2*Rd;
    end

    % Terminal cost
    dxN = X(:,N) - nominal(:,N);
    dxN(3) = wrapToPi(dxN(3));
    lx(:,N)    = 2*Qf*dxN;
    lxx(:,:,N) = 2*Qf;
end

function [K,k,ok,reg] = backward_pass(A,B,lx,lu,lxx,luu,lux,reg)
    nx=size(A,1); nu=size(B,2); N=size(A,3)+1;
    Vx=lx(:,N); Vxx=lxx(:,:,N);
    K=zeros(nu,nx,N-1); k=zeros(nu,N-1);
    ok=true;
    for t=N-1:-1:1
        At=A(:,:,t); Bt=B(:,:,t);
        Qx  = lx(:,t)+At'*Vx;
        Qu  = lu(:,t)+Bt'*Vx;
        Qxx = lxx(:,:,t)+At'*Vxx*At;
        Quu = luu(:,:,t)+Bt'*Vxx*Bt;
        Qux = lux(:,:,t)+Bt'*Vxx*At;

        % Levenberg–Marquardt regularization
        while true
            Quu_reg = Quu + reg*eye(nu);
            [~,p] = chol(Quu_reg);
            if p==0, break; end
            reg = reg*10;
            if reg > 1e6
                ok=false; return;
            end
        end
        invQuu = inv(Quu_reg);
        k(:,t)   = -invQuu*Qu;
        K(:,:,t) = -invQuu*Qux;

        Vx  = Qx + K(:,:,t)'*Quu*k(:,t) + K(:,:,t)'*Qu + Qux'*k(:,t);
        Vxx = Qxx+ K(:,:,t)'*Quu*K(:,:,t) + K(:,:,t)'*Qux + Qux'*K(:,:,t);
        Vxx = 0.5*(Vxx+Vxx');   % symmetrize
    end
end

function [Xbest,Ubest,Jbest,success] = forward_search( ...
        x0,X,U,K,k,alpha_list,dt,vmax,wmax, ...
        nominal,Q,R,Qf,Rd, ...
        dist_map,xlim,ylim,res,w_obs,sigma_obs)

    success=false;
    Jbest = total_cost(X,U,nominal,Q,R,Qf,Rd, ...
                       dist_map,xlim,ylim,res,w_obs,sigma_obs);
    Xbest=X; Ubest=U;

    for a = 1:length(alpha_list)
        alpha = alpha_list(a);
        Xc=zeros(size(X)); Uc=zeros(size(U));
        Xc(:,1)=x0;
        for t=1:size(U,2)
            du = alpha*k(:,t)+K(:,:,t)*(Xc(:,t)-X(:,t));
            unew = U(:,t)+du;
            % clip
            unew(1)=max(-vmax,min(vmax,unew(1)));
            unew(2)=max(-wmax,min(wmax,unew(2)));
            Uc(:,t)=unew;
            Xc(:,t+1)=dynamics_nominal(Xc(:,t),unew,dt);
        end
        Jc = total_cost(Xc,Uc,nominal,Q,R,Qf,Rd, ...
                        dist_map,xlim,ylim,res,w_obs,sigma_obs);
        if Jc < Jbest
            Jbest=Jc; Xbest=Xc; Ubest=Uc; success=true; break;
        end
    end
end

function J = total_cost(X,U,nominal,Q,R,Qf,Rd, ...
                         dist_map,xlim,ylim,res,w_obs,sigma_obs)
    N=size(X,2); J=0;
    for t=1:N-1
        dx = X(:,t)-nominal(:,t);
        dx(3) = wrapToPi(dx(3));
        % state cost
        J = J + dx'*Q*dx;
        % obstacle cost
        c_obs = obstacle_cost(X(:,t),dist_map,xlim,ylim,res,w_obs,sigma_obs);
        J = J + c_obs;
        % control effort
        J = J + U(:,t)'*R*U(:,t);
        % control smoothing
        if t == 1
            u_prev = [0;0];
        else
            u_prev = U(:,t-1);
        end
        du = U(:,t) - u_prev;
        J = J + du'*Rd*du;
    end
    dxN = X(:,N)-nominal(:,N);
    dxN(3) = wrapToPi(dxN(3));
    J=J+dxN'*Qf*dxN;
end

function [A,B] = linearize_dynamics(x,u,dt)
    th=x(3); v=u(1);
    A=eye(3);
    A(1,3)=-v*sin(th)*dt;
    A(2,3)= v*cos(th)*dt;
    B=zeros(3,2);
    B(1,1)=cos(th)*dt;
    B(2,1)=sin(th)*dt;
    B(3,2)=dt;
end

%% PID BASELINE WITH SAME MISMATCH + NOISE

function [X,U] = baseline_pid_with_noise(x0, nominal, dt, vmax,wmax, mismatch, noise_seq)
    N = size(nominal,2);
    total_steps = N-1;
    X = zeros(3,N); U = zeros(2,total_steps);
    X(:,1) = x0;

    % Actuator state for PID plant
    act.v = 0; act.w = 0;

    Kp_v=0.8; Kp_w=3.0;
    for t=1:total_steps
        dx = nominal(1:2,t)-X(1:2,t);
        dist = norm(dx);
        desired = atan2(dx(2),dx(1));
        herr = wrapToPi(desired-X(3,t));
        v = Kp_v*dist;
        w = Kp_w*herr;
        v=max(-vmax,min(vmax,v));
        w=max(-wmax,min(wmax,w));
        U(:,t)=[v;w];

        % true plant: mismatch + actuator lag + SAME noise sequence
        [x_nom, act] = dynamics_plant(X(:,t), U(:,t), dt, mismatch, act);
        xnext = x_nom + noise_seq(:,t);
        xnext(3) = wrapToPi(xnext(3));
        X(:,t+1) = xnext;
    end
end

%% DYNAMICS (NOMINAL vs TRUE PLANT)

function xnext = dynamics_nominal(x,u,dt)
    % Nominal simple kinematic dynamics (used inside iLQR)
    xnext=zeros(3,1);
    th=x(3);
    xnext(1)=x(1)+u(1)*cos(th)*dt;
    xnext(2)=x(2)+u(1)*sin(th)*dt;
    xnext(3)=wrapToPi(x(3)+u(2)*dt);
end

function [xnext, act_out] = dynamics_plant(x,u_cmd,dt,mismatch,act_in)
    % True plant dynamics with:
    %   - velocity scaling mismatch
    %   - actuator lag
    %   - heading bias
    if nargin<5 || isempty(act_in)
        act_in.v = 0;
        act_in.w = 0;
    end
    act_out = act_in;

    v_cmd = mismatch.v_scale * u_cmd(1);
    w_cmd = u_cmd(2) + mismatch.w_bias;    % heading bias

    % First-order actuator lag
    alpha_v = dt / max(mismatch.tau_v, eps);
    alpha_w = dt / max(mismatch.tau_w, eps);
    act_out.v = act_out.v + alpha_v * (v_cmd - act_out.v);
    act_out.w = act_out.w + alpha_w * (w_cmd - act_out.w);

    v = act_out.v;
    w = act_out.w;
    dt_eff = mismatch.dt_scale * dt;

    xnext=zeros(3,1);
    th=x(3);
    xnext(1)=x(1)+v*cos(th)*dt_eff;
    xnext(2)=x(2)+v*sin(th)*dt_eff;
    xnext(3)=wrapToPi(x(3)+w*dt_eff);
end

function y = wrapToPi(x)
    y = mod(x+pi,2*pi)-pi;
end

%% OBSTACLE COST & GRADIENT HELPERS

function c_obs = obstacle_cost(x,dist_map,xlim,ylim,res,w_obs,sigma)
    [d, ~, ~] = distance_with_grad(x(1),x(2),dist_map,xlim,ylim,res);
    d = max(d, 0);   % ensures non-negative
    c_obs = w_obs * exp(-d/sigma);
end

function [c_obs, grad] = obstacle_cost_grad(x,dist_map,xlim,ylim,res,w_obs,sigma)
    [d, ddx, ddy] = distance_with_grad(x(1),x(2),dist_map,xlim,ylim,res);
    d = max(d, 0);
    base  = w_obs * exp(-d/sigma);
    c_obs = base;
    if d == 0
        grad = [0;0;0];  % inside obstacle or right at boundary; relying on Q
        return;
    end
    coeff = -base / sigma;
    grad  = [coeff*ddx; coeff*ddy; 0];
end

function [d, ddx, ddy] = distance_with_grad(x,y,dist_map,xlim,ylim,res)
    [ny,nx] = size(dist_map);

    ix = round((x - (xlim(1)+res/2))/res) + 1;
    iy = round((y - (ylim(1)+res/2))/res) + 1;

    ix = max(1,min(nx,ix));
    iy = max(1,min(ny,iy));

    d = dist_map(iy,ix);

    % Approximate gradient via central differences
    ixp = min(nx, ix+1);
    ixm = max(1,  ix-1);
    iyp = min(ny, iy+1);
    iym = max(1,  iy-1);

    dx_plus  = dist_map(iy,ixp);
    dx_minus = dist_map(iy,ixm);
    dy_plus  = dist_map(iyp,ix);
    dy_minus = dist_map(iym,ix);

    ddx = (dx_plus - dx_minus) / (2*res);
    ddy = (dy_plus - dy_minus) / (2*res);
end

%% ANIMATION & FINAL FIGURES

function animate_and_finalfig(occ,occ_inf,res,xlim,ylim, ...
        nominal,X_mpc,U_mpc,X_pid,U_pid,start,goal,dt,robot_r, ...
        cost_mpc, pos_err_mpc, pos_err_pid)

    nx=size(occ,2); ny=size(occ,1);
    xs=linspace(xlim(1)+res/2,xlim(2)-res/2,nx);
    ys=linspace(ylim(1)+res/2,ylim(2)-res/2,ny);

    hf=figure('Name','MPC iLQR vs PID (Hybrid A* with Noise/Mismatch)','Position',[150 120 1200 700]);
    ax=subplot(2,3,[1 4]); hold(ax,'on'); axis(ax,'equal');
    imagesc(xs,ys,flipud(occ)); colormap(ax,gray); set(ax,'YDir','normal');
    % Slight transparency overlay for inflated obstacles
    occ_inf_img = flipud(occ_inf);
    h_occinf = imagesc(xs,ys,occ_inf_img);
    set(h_occinf,'AlphaData',0.2); colormap(ax,gray);
    plot(ax,nominal(1,:),nominal(2,:),'y--','LineWidth',1.2,'DisplayName','Nominal');
    h_mpc = plot(ax,NaN,NaN,'b-','LineWidth',2,'DisplayName','MPC iLQR');
    h_pid = plot(ax,NaN,NaN,'r-','LineWidth',1.5,'DisplayName','PID');
    plot(ax,start(1),start(2),'mo','MarkerFaceColor','m','DisplayName','Start');
    plot(ax,goal(1),goal(2),'cx','MarkerSize',10,'LineWidth',2,'DisplayName','Goal');
    legend(ax,'Location','eastoutside');
    xlabel(ax,'x (m)'); ylabel(ax,'y (m)');

    hp_mpc = patch(ax,NaN,NaN,'b','FaceAlpha',0.4,'EdgeColor','k');
    hp_pid = patch(ax,NaN,NaN,'r','FaceAlpha',0.4,'EdgeColor','k');

    ax_v=subplot(2,3,2); hold(ax_v,'on'); title(ax_v,'Linear v (m/s)'); xlabel(ax_v,'time (s)');
    ax_w=subplot(2,3,5); hold(ax_w,'on'); title(ax_w,'Angular \omega (rad/s)'); xlabel(ax_w,'time (s)');

    tvec=(0:(size(U_mpc,2)-1))*dt;
    plot(ax_v,tvec,U_mpc(1,:),'b-','LineWidth',1.2,'DisplayName','MPC iLQR');
    plot(ax_v,tvec,U_pid(1,1:length(tvec)),'r--','DisplayName','PID');
    legend(ax_v,'Location','best'); grid(ax_v,'on');

    plot(ax_w,tvec,U_mpc(2,:),'b-','LineWidth',1.2,'DisplayName','MPC iLQR');
    plot(ax_w,tvec,U_pid(2,1:length(tvec)),'r--','DisplayName','PID');
    legend(ax_w,'Location','best'); grid(ax_w,'on');

    ax_e = subplot(2,3,3); hold(ax_e,'on');
    plot(ax_e, tvec, pos_err_mpc(1:length(tvec)), 'b-', 'LineWidth',1.2, 'DisplayName','MPC iLQR');
    plot(ax_e, tvec, pos_err_pid(1:length(tvec)), 'r--','LineWidth',1.2,'DisplayName','PID');
    title(ax_e,'Position error ||x - x_{nom}||'); xlabel(ax_e,'time (s)');
    ylabel(ax_e,'error (m)'); legend(ax_e,'Location','best'); grid(ax_e,'on');

    Nsteps=size(X_mpc,2);
    for k=1:Nsteps
        set(h_mpc,'XData',X_mpc(1,1:k),'YData',X_mpc(2,1:k));
        set(h_pid,'XData',X_pid(1,1:k),'YData',X_pid(2,1:k));
        update_triangle(hp_mpc,X_mpc(:,k),robot_r);
        update_triangle(hp_pid,X_pid(:,k),robot_r);
        title(ax,sprintf('Time %.2f s (%d/%d)',(k-1)*dt,k,Nsteps));
        drawnow;
        pause(0.01);
    end

    % Final static summary figure: trajectories + MPC horizon cost
    hf2=figure('Name','Final Trajectories + MPC Horizon Cost','Position',[200 140 1200 700]);

    a1=subplot(2,2,[1 3]); hold(a1,'on'); axis(a1,'equal');
    imagesc(xs,ys,flipud(occ)); colormap(a1,gray); set(a1,'YDir','normal');
    occ_inf_img2 = flipud(occ_inf);
    h2 = imagesc(xs,ys,occ_inf_img2); set(h2,'AlphaData',0.2);
    plot(a1,nominal(1,:),nominal(2,:),'y--','LineWidth',1.2);
    plot(a1,X_mpc(1,:),X_mpc(2,:),'b-','LineWidth',2);
    plot(a1,X_pid(1,:),X_pid(2,:),'r-','LineWidth',1.5);
    plot(a1,start(1),start(2),'mo','MarkerFaceColor','m');
    plot(a1,goal(1),goal(2),'cx','MarkerSize',10,'LineWidth',2);
    legend(a1,'Nominal','MPC iLQR','PID','Start','Goal','Location','eastoutside');
    title(a1,'Final trajectories (with noise/mismatch)');

    a2=subplot(2,2,2); hold(a2,'on');
    plot(a2,tvec,U_mpc(1,:),'b-','LineWidth',1.2);
    plot(a2,tvec,U_pid(1,1:length(tvec)),'r--','LineWidth',1.2);
    title(a2,'Linear velocity v'); xlabel(a2,'time (s)');
    grid(a2,'on'); legend(a2,'MPC iLQR','PID','Location','best');

    a3=subplot(2,2,4); hold(a3,'on');
    plot(a3,1:numel(cost_mpc),cost_mpc,'g-o','LineWidth',1.2);
    title(a3,'MPC iLQR horizon cost per step');
    xlabel(a3,'MPC step'); ylabel(a3,'cost'); grid(a3,'on');

    saveas(hf2,'final_summary_ilqr_mpc_noise.png');
    fprintf('Saved final static figure as final_summary_ilqr_mpc_noise.png\n');
end

function update_triangle(hpatch,x,r)
    th=x(3);
    pts=[r,0; -0.6*r,0.4*r; -0.6*r,-0.4*r]';
    Rm=[cos(th) -sin(th); sin(th) cos(th)];
    rot=Rm*pts;
    Xp=x(1)+rot(1,:);
    Yp=x(2)+rot(2,:);
    set(hpatch,'XData',Xp,'YData',Yp);
end