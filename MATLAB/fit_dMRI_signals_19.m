%% Sub-diffusion model fitting (Delta = 19 ms)
clear;
SUBJECT  = 'sub_001';
datatype = '_real'; % '_real' or empty string ''
FOLDER   = 'C:\Users\adyna\Downloads\dwi_real\dwi_real\';

bvec = [0 50 350 800 1500 2400 3450 4750 6000]'; % in units of s/mm^2
nb   = length(bvec);

%% Load data
brainmask = double(niftiread([FOLDER SUBJECT '_dwi' datatype '_brainmask.nii.gz']));

% S(:,:,:,k) is the image for bvec(k)
S0 = double(niftiread([FOLDER SUBJECT '_dwi' datatype '_b0_delta0_image.nii'])).*brainmask;
[X,Y,L] = size(S0);
S = zeros(X,Y,L,nb);
S(:,:,:,1) = S0;
for k = 2:nb
    S(:,:,:,k) = double(niftiread([FOLDER SUBJECT '_dwi' datatype '_b' num2str(bvec(k)) '_delta19_image.nii'])).*brainmask;
end

%% Fit model parameters
% Sub-diffusion model: D and beta, computing K* from beta

D_guess    = 1e-3; Dmin    = 0; Dmax    = 3e-3; % mm^2/s
beta_guess = 0.5;  betamin = 0; betamax = 1;    % time fractional index

D_SUB     = nan(X,Y,L);
BETA_SUB  = nan(X,Y,L);
error_SUB = nan(X,Y,L);

num_voxels = nnz(S0 > 0);
fprintf('Fitting %d brain voxels\n', num_voxels);

tic
for l = 1:L % whole brain
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
fprintf('Fit time: %.1f s (%.3f ms/voxel)\n', fit_time, 1e3 * fit_time / num_voxels);

K_STAR = 6 .* gamma(1+BETA_SUB).^2 ./ gamma(1+2*BETA_SUB) - 3;

%% Parameter maps
l = 73; % axial slice to display

figure; t = tiledlayout(1,4,'TileSpacing','tight','Padding','compact');
title(t,'Parameter maps','Color',[1,1,1])
set(gcf, 'Color', [0 0 0]);

nexttile
imagesc(squeeze(D_SUB(:,:,l))), axis image, title('D_{SUB}','Color',[1,1,1]); clim([0 3e-3]);
colorbar('Color',[1,1,1]); view(-90,90); axis off; colormap hot

nexttile
imagesc(squeeze(error_SUB(:,:,l))), axis image, clim([0 0.1]); title('RMSE','Color',[1,1,1]);
colorbar('Color',[1,1,1]); view(-90,90); axis off;

nexttile
imagesc(squeeze(BETA_SUB(:,:,l))), axis image, clim([0.5 1]); title('Beta','Color',[1,1,1]);
colorbar('Color',[1,1,1]); view(-90,90); axis off;

nexttile
imagesc(squeeze(K_STAR(:,:,l))), axis image, clim([0 3]); title('K^*','Color',[1,1,1]);
colorbar('Color',[1,1,1]); view(-90,90); axis off;

%% Check fit at a single voxel
x = 43; y = 55; l = 73;

S_measured = squeeze(S(x,y,l,:));
b_fine     = linspace(0, max(bvec), 200);
S_fit      = S0(x,y,l) * ml(-b_fine * D_SUB(x,y,l), BETA_SUB(x,y,l));

figure('Color', 'w');
hold on;
plot(bvec / 1000, S_measured, 'o', 'Color', [0 0.4470 0.7410], ...
    'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'data');
plot(b_fine / 1000, S_fit, '-', 'Color', [0.4940 0.1840 0.5560], ...
    'LineWidth', 2, 'DisplayName', 'Sub-diffusion');

xlabel('b-value (\times 10^3 s/mm^2)', 'FontWeight', 'bold');
ylabel('S', 'FontWeight', 'bold');
box on; grid off;
ylim([0, max(S_measured) * 1.05]);
set(gca, 'FontSize', 11, 'LineWidth', 1);
legend('Location', 'northeast');
