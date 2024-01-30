% This is a demo script on how to use the OCT Simulator to generate a
% B-scan image from a synthetic object using OCT imaging model
% The script is composed of
% 1. Defintiion of OCT imaging system parameters
% 2. Definition of sample properties and structure
% 3. Sample generation
% 4. Simulation of the OCT tomogram
% 5. Calculation of Stokes vectors and DOP
% 6. Visualization

%% Add path with functions
addpath(genpath('../matlab'));
% Some options for figures
LATEX_DEF = {'Interpreter', 'latex'};
set(0, 'defaultTextInterpreter', 'LaTex')
set(0, 'defaultAxesTickLabelInterpreter', 'LaTex')
set(groot,'defaultLegendInterpreter','latex');
set(0, 'DefaultAxesFontSize', 20)
% Use fixed seed for random generation if interested in repeated trials
% Comment otherwise
rng(123);

%% 1. Parameters of the OCT imaging system
% Number of samples
nZ = 512; % axial direction, number of pixels per A-line
nX = 512;  % fast scan axis, number of A-lines per B-scan
nY = 1;    % slow scan axis, number of B-scans per tomogram
nK = 512;  % number of k-samples, must be <= nZ, difference is the size of zero padding
xNyquistOversampling = 1; % Lateral sampling factor. 1 -> Nyquist sampling

% Spectral parameters
centralWavelength = 1.040e-6; % source central wavelength in m
wavelengthRange = 110e-9;     % Full spectral bandwidth in m
% Spectral windows
nWindows = 1; % Use one window for full resolution tomogram or multiple windows for spectral binning reconstruction
if nWindows > 1
  specOverlap = 2 / 3;
  spectralWindows = permute(GenerateMovingSpectralWindow(nK, specOverlap, nWindows), [1 6 3 4 5 2]);
else
  spectralWindows = hanning(nK);
end
% Flag to enable confocal function
confocalFuncOn = true;

% Numerical aperture
numAper = 0.05;
% Flag to enable sensitivity fall-off
sensitivityOn = true;
% Noise floor level in dB
noiseFloorDb = 50; % Actual noisefloor in the final tomogram is affected by spectral windowing
% Index of refraction of the media
mediaIndex = (4 / 3); % dimensionless

% Polarization-related parameters
% Flag used to do polarization-diverse simulation
% True -> the scattering potential is a matrix
% Flase -> the scattering potential is a scalar
doPS = true;
% Sample input polarization state, concatenate along 2nd dimension as many
% input states as desired. Two polarization-diverse signals will be
% concatenated along the 5th dimension is the simulated tomogram for each
% input polarization state
% Let's try linear and +45
inputPolState = cat(2, 1 / sqrt(1) * [1; 0], 1 / sqrt(2) * [1; -1]);
% Flag to add Differential group delay dispersion (DGDD) also known as
% polarization mode dispersion (PMD)
addDGDD = false;
if addDGDD % Add DGDD
  jDGDDretRange = [0 2*pi/3;... % Range of DGDD for system matrix J_A
                   pi/8 2*pi/4]; % Range of DGDD for system matrix J_B
  jDGDDangle = [pi/4,... % Orientation of DGDD for system matrix J_A
                2*pi/3]; % Orientation of DGDD for system matrix J_B
else % Not dot add DGDD
  jDGDDretRange = [0 0;... % Range of DGDD for system matrix J_A
                   0 0]; % Range of DGDD for system matrix J_B
  jDGDDangle = [0,... % Orientation of DGDD for system matrix J_A
                0]; % Orientation of DGDD for system matrix J_B
end

% Flag to use efficient estimation, meaning that the scatterers located 
% too far from the current spot location are ignored, because their
% contribution is negligible. Use False when having sparse scatterers
efficient = true;
% Flag to use GPU
useGPU = true;
% Make variables gpuArray is using GPU
if useGPU; varType = {'single', 'gpuArray'}; else; varType = {'single'}; end

% Create struct with input parameters
parms = struct('tomSize', [nZ, nX, nY], 'nK', nK, 'centralWavelength',...
  centralWavelength, 'wavelengthRange', wavelengthRange, 'numAper', numAper,...
  'xNyquistOversampling', xNyquistOversampling, 'noiseFloorDb', noiseFloorDb,...
  'useGPU', useGPU, 'plotPSF', 20, 'efficient', efficient, 'mediaIndex', mediaIndex,...
  'confocalFuncOn', confocalFuncOn, 'sensitivityOn', sensitivityOn, 'verbosity', 2,...
  'inputPolState', inputPolState, 'spectralWindows', spectralWindows,...
  'jDGDDretRange', jDGDDretRange, 'jDGDDangle', jDGDDangle, 'doPS', doPS);
% Compute simulation parameters
simParms = GetTomogramParameters(parms);
StructToVars(simParms);

%% 2. Definition of sample properties and structure
% Scatterers density threshold for fully developed speckle
fdsThresh = 8 / gather(resolutionVol); % 8 Scatterers per resolution volume
% Sample range in m. This is the max range but it can be set smaller.
objRangeXY = [nX, nY] .* [latSampling, latSampling];
% Sample axial length in px
objZRange = 150;
% Number of layers
nLayers = 5;

% Layers properties
% Thickness
layerThickness =  [1/5 1/5 1/5 1/5 1/5]; % Easier to define ratios then convert to distane units
layerThickness = single(layerThickness / sum(layerThickness) * objZRange * axSampling); % in m
% Thickness irregularity -> controls layer flatness
layerIrregularity = [1e-5 0 2e-5 0 2e-5 1e-5]; % Scalar (same for all layers) or vector (one for each layer)
% Scatterers density per layer and per constituent (layers can have
% multiple constituents)
scatterersDensity = single([1 0; 1 0; 1/4 3/4; 5/4 0; 1 0]') * fdsThresh; % in scatterers per m^-3
% Radius of each constituent in each layer
scaterrersRadius = [1.5 1.0; 1.3 1.0; 1.7 1.1; 1.8 1.0; 1.3 1.0]' * centralWavelength / 10; % in m
% Index of refraction of each constituent in each layer
scatterersIndex = [1.5 1.5; 1.5 1.5; 1.5 1.5; 1.5 1.5; 1.5 1.5]'; % dimensionless
% Absoption coefficient of each layer
absoptionCoef = ones(1, nLayers) * 1e2; % in m^-1;

% Polarization-related properties
% Birefringence
layersBirefringence = [5 0 10 0 0] * 1e-4;  % Dimensionless
% Orientation of the fast axis
layersFastAxisAngle = [pi / 3 0 pi / 4 0 0] * pi / 1; % radians
% Depolarization in the backward-scattering direction
layerBackwardDepolarization = [0.0 0.2 0.0 0.0 0.0];  % Dimensionless
% Depolarization in the forward-scattering direction
layerForwardDepolarization = [0.0 0.0 0.0 0.3 0.0];  % Dimensionles
% Create struct with sample properties
objParms = struct('objRangeXY', objRangeXY, 'layerThickness', layerThickness,...
  'scatterersDensity', scatterersDensity, 'scaterrersRadius', scaterrersRadius,...
  'scatterersIndex', scatterersIndex, 'mediaIndex', mediaIndex,...
  'latSampling', [latSampling,  latSampling], 'axSampling', axSampling,...
  'centralWavelength', centralWavelength,...
  'centralWavenumber', simParms.centralWavenumber,...
  'wavenumberVect', wavenumberVect, 'varType', {varType},...
  'absoptionCoef', absoptionCoef, 'numAper', numAper,...
  'maxBatchSize', [], 'structreImgSize', [nZ, nX, nY],...
  'layerIrregularity', layerIrregularity, 'fdsThresh', fdsThresh,...
  'doPS', doPS, 'layersBirefringence', layersBirefringence,...
  'layersFastAxisAngle', layersFastAxisAngle,...
  'layerForwardDepolarization', layerForwardDepolarization,...
  'layerBackwardDepolarization', layerBackwardDepolarization,...
  'inputPolState', inputPolState);

%% 3. Sample generation
[objPos, objBackscat, objLayers, objLayerLabels,...
  backScatProfile, backScatCoefProfile] = ...
  CreateIrregularLayeredBirefringentSampleRayleigh(objParms);

%% 4. Simulation of the OCT tomogram
% Additional parameteres
% Number of points per constituent per layer
nPointSources = sum(round(sum(layerThickness .* objRangeXY(1) .*...
  objRangeXY(2) .* scatterersDensity, 1)));
parms.maxPointsBatch = min(nPointSources, 1e5);
% Focal plane location (w.r.t. reference mirror path-length)
parms.focalPlane = 5e-4;
% Reference mirror location (w.r.t. sample's zero)
parms.zRef = -1e-4;
StructToVars(parms);
% Simulate tomogram
[tom, ~, simParms] = SimulateTomogram(objBackscat, objPos, parms);
if useGPU; tom = gather(tom); end

%% 5. Calculation of Stokes vectors and DOP
% Convert Jones vectors into Stokes-Jones vectors
StokesIndx = 4;
tomStokes = ComplexTomToStokes(tom, StokesIndx);
% Re-order dimensions (only if nY = 1)
if size(tom, 3) == 1; tomStokes = permute(tomStokes, [1 2 6 4 3 5]); end
% Define window for averaging Stokes vectors
windowHalfSizeX = 21;
filtWindowX = AnisotropicGaussianExp2Diameter([floor(windowHalfSizeX) * 2 + 1, 1],...
  windowHalfSizeX, 1);
% Average the Stokes vectors
if useGPU
  tomStokesAve = gather(imfilter(gpuArray(tomStokes), filtWindowX, 'symmetric'));
else
  tomStokesAve = imfilter(tomStokes, filtWindowX, 'symmetric');
end
% compute DOP
tomDOP = mean(vecnorm(tomStokesAve(:, :, :, 2:4, :), 2, StokesIndx) ./ ...
  tomStokesAve(:, :, :, 1, :), 5);

%% 6. Visualization
% Display options
% Limits for intensity image, in dB
logLim = [noiseFloorDb-5 inf];
% Limits for DOP
dopRange = [0 1];
% Colormap for images
grayCMap = gray(256);
% Offset for figure window number
curFig = 0;
% B-scan number to display
thisBscan = 1;

% Sample backscattering
figure(curFig + 1); subplot(1,3,1), imagesc(squeeze(xVect) * 1e3, ...
  (squeeze(zVect) + 0*depthRange) * 1e3,... %
  backScatCoefProfile(1:end/2, :, thisBscan)), axis image, colormap(grayCMap),
xlabel('$x$ [$\mu$m]'), ylabel('$z$ [$\mu$m]'), title('(a) Sample B-scan'),
yline(focalPlane * 1e3, 'r--', 'linewidth', 2), yline(zRef * 1e3, 'b--', 'linewidth', 2),
% Total intensity for input polarization state 1
figure(curFig + 1); subplot(1,3,2), imagesc(squeeze(xVect) * 1e3, squeeze(zVect)* 1e3,...
  10 * log10(sum(abs(tom(:, :, thisBscan, :, 2)), 4) .^ 2), logLim), set(gca, 'YDir','reverse')
xlabel('$x$ [$\mu$m]'), title('(b) OCT B-scan image'), axis image % ylabel('$z$ [$\mu$m]'),
yline((focalPlane - zRef) * 1e3, 'r--', 'linewidth', 2), yline(0, 'b--', 'linewidth', 2),
% Total intensity for input polarization state 2
figure(curFig + 1); subplot(1,3,3), imagesc(squeeze(xVect) * 1e3, squeeze(zVect)* 1e3,...
  10 * log10(sum(abs(tom(:, :, thisBscan, :, 2)), 4) .^ 2), logLim), set(gca, 'YDir','reverse')
xlabel('$x$ [$\mu$m]'), title('(b) OCT B-scan image'), axis image % ylabel('$z$ [$\mu$m]'),
yline((focalPlane - zRef) * 1e3, 'r--', 'linewidth', 2), yline(0, 'b--', 'linewidth', 2),

% Independent channels intensity
% Input state 1, horizontal component
figure(curFig + 2), subplot(1, 4, 1), imagesc(10 * log10(abs(tom(:, :, thisBscan, 1, 1, 1)) .^ 2), logLim),
axis image, colormap(gray(256)), axis off, title('$I_x^{(1)}$')
% Input state 1, vertical component
figure(curFig + 2), subplot(1, 4, 2), imagesc(10 * log10(abs(tom(:, :, thisBscan, 2, 1, 1)) .^ 2), logLim),
axis image, colormap(gray(256)), axis off, title('$I_y^{(1)}$')
% Input state 2, horizontal component
figure(curFig + 2), subplot(1, 4, 3), imagesc(10 * log10(abs(tom(:, :, thisBscan, 1, 2, 1)) .^ 2), logLim),
axis image, colormap(gray(256)), axis off, title('$I_x^{(2)}$')
% Input state 2, vertical component
figure(curFig + 2), subplot(1, 4, 4), imagesc(10 * log10(abs(tom(:, :, thisBscan, 2, 2, 1)) .^ 2), logLim),
axis image, colormap(gray(256)), axis off, title('$I_y^{(2)}$')

% Display Stokes vectors
% Show intensities and QUV for all input polarization states and spectral bins
nPolIn = size(tom, 5);
for thisWin = 1:nWindows
  for thisPolIn = 1:nPolIn
    % Intensity in the first detection channel
    figure(curFig + 3 + thisWin - 1), splotH = subplot(2, 6, 6 * (thisPolIn - 1) + 1);
    imagesc(10 * log10(abs(tom(:, :, thisBscan, 1, thisPolIn, thisWin) .^ 2)), logLim),
    axis image, splotH.Colormap = gray(256); axis off, title(sprintf('$I_x^{(%d)}$', thisPolIn))
    % Intensity in the second detection channel
    splotH = subplot(2, 6, 6 * (thisPolIn - 1) + 2);
    imagesc(10 * log10(abs(tom(:, :, thisBscan, 2, thisPolIn, thisWin) .^ 2)), logLim),
    axis image, splotH.Colormap = gray(256); axis off, title(sprintf('$I_y^{(%d)}$', thisPolIn))
    % Total intensity (I component of Stokes vectors)
    splotH = subplot(2, 6, 6 * (thisPolIn - 1) + 3);
    imagesc(10 * log10(tomStokes(:, :, thisBscan, 1, thisPolIn, thisWin)), logLim),
    axis image, splotH.Colormap = gray(256); axis off, title(sprintf('$I^{(%d)}$', thisPolIn))
    % Normalized QUV components of Stokes vectors
    splotH = subplot(2, 6, 6 * (thisPolIn - 1) + [4 5 6]);
    imagesc(FlattenArrayTo2D(tomStokes(:, :, thisBscan, 2:4, thisPolIn, thisWin) ./...
      tomStokes(:, :, thisBscan, 1, thisPolIn, thisWin)), [-1 1]),
    axis image, splotH.Colormap = gray(256); axis off, title('$(Q,\ U,$ and $V) / I$')
  end
end

% Display DOP
figure(curFig + 4 + (nWindows - 1)), imagesc(tomDOP(:, :, thisBscan), dopRange)
colormap(viridis(256)); axis image, colorbar, title('DOP')

