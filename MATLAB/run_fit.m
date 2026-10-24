function run_fit(slice, out_dir)
% RUN_FIT  The original sub-diffusion fit (fit_dMRI_signals_19.m), for timing and comparison.
%
%   run_fit()            fits every brain voxel
%   run_fit(73)          fits only axial slice 73
%   run_fit(73, 'MATLAB/outputs')
%
% Same data loading, loop order and fit_voxel_sub_ml calls as fit_dMRI_signals_19.m, but
% writes the maps as NIfTI plus run.txt to <out_dir>/all or <out_dir>/slice_N, in the same
% layout as the C++ and CUDA versions so compare/compare_results.py can read them all.
% Run from the repository root (data in ./data/dwi_real/). Needs the Image Processing
% Toolbox (niftiread) and the Optimization Toolbox (lsqcurvefit).

if nargin < 1, slice = []; end
if nargin < 2, out_dir = fullfile('MATLAB', 'outputs'); end

SUBJECT  = 'sub_001';
datatype = '_real';
FOLDER   = fullfile('data', 'dwi_real', filesep);

bvec = [0 50 350 800 1500 2400 3450 4750 6000]'; % in units of s/mm^2
nb   = length(bvec);

%% Load data (as in fit_dMRI_signals_19.m)
brainmask = double(niftiread([FOLDER SUBJECT '_dwi' datatype '_brainmask.nii.gz']));

b0_file = [FOLDER SUBJECT '_dwi' datatype '_b0_delta0_image.nii'];
S0 = double(niftiread(b0_file)).*brainmask;
[X,Y,L] = size(S0);
S = zeros(X,Y,L,nb);
S(:,:,:,1) = S0;
for k = 2:nb
    S(:,:,:,k) = double(niftiread([FOLDER SUBJECT '_dwi' datatype '_b' num2str(bvec(k)) '_delta19_image.nii'])).*brainmask;
end

%% Fit model parameters (as in fit_dMRI_signals_19.m, optionally one slice)
D_guess    = 1e-3; Dmin    = 0; Dmax    = 3e-3; % mm^2/s
beta_guess = 0.5;  betamin = 0; betamax = 1;    % time fractional index

D_SUB     = nan(X,Y,L);
BETA_SUB  = nan(X,Y,L);
error_SUB = nan(X,Y,L);

if isempty(slice)
    slices = 1:L; run_name = 'all';
else
    slices = slice;  run_name = sprintf('slice_%d', slice);
end

num_voxels = nnz(S0(:,:,slices) > 0);
fprintf('Variant: matlab\nFitting %d brain voxels (%s)\n', num_voxels, run_name);

tic
for l = slices
    for x = 1:X
        for y = 1:Y
            % Skip voxels outside brain mask
            if S0(x,y,l) <= 0
                continue;
            end

            Svec = squeeze(S(x,y,l,:)) / S0(x,y,l);

            [D_SUB(x,y,l), BETA_SUB(x,y,l), error_SUB(x,y,l)] = fit_voxel_sub_ml(Svec, bvec, D_guess, beta_guess, Dmin, Dmax, betamin, betamax);
        end
    end
end
fit_time = toc;
fprintf('Fit time: %.3f s (%.4f ms/voxel)\n', fit_time, 1e3 * fit_time / num_voxels);

K_STAR = 6 .* gamma(1+BETA_SUB).^2 ./ gamma(1+2*BETA_SUB) - 3;

%% Save results: NIfTI with the input's geometry, plus run details
save_path = fullfile(out_dir, run_name);
if ~exist(save_path, 'dir'), mkdir(save_path); end

info = niftiinfo(b0_file);
info.Datatype = 'double';
info.BitsPerPixel = 64;
info.MultiplicativeScaling = 1;
info.AdditiveOffset = 0;
niftiwrite(D_SUB,     fullfile(save_path, 'D_SUB.nii'),     info);
niftiwrite(BETA_SUB,  fullfile(save_path, 'BETA_SUB.nii'),  info);
niftiwrite(error_SUB, fullfile(save_path, 'ERROR_SUB.nii'), info);
niftiwrite(K_STAR,    fullfile(save_path, 'K_STAR.nii'),    info);

fid = fopen(fullfile(save_path, 'run.txt'), 'w');
fprintf(fid, 'variant=matlab\nthreads=1\nvoxels=%d\nfit_time_s=%.6f\n', num_voxels, fit_time);
fclose(fid);
fprintf('Saved results to %s\n', save_path);
end
