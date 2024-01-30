function simParms = GetTomogramParameters(inputParms)

% GetTomogramParameters Creates OCT simulation parameteres given input set
% of parameters
%
% simParms = GetTomogramParameters(inputParms)
%
% Inputs:
% scattAmp: Sample's scattering potential
% scattPos: Z, X, and Y position of point scatterers in m.
% inputParms: struct with optinal inputs:
%   centralWavelength: Central wavelength.
%   wavelengthRange: Full spectral bandwidth.
%   numAper: Numerical aperture.
%   xNyquistOversampling: Factor to oversample laterally (1-> Nyquist).
%   noiseFloorDb: Noise level in dB, added to the fringes.
%   focalPlane: Location of the focal plane wrt to the sample in m.
%   zRef: Location of zero-path delay wrt sample in m.
%   useGPU: Use GPU to evaluate the forward model.
%   centralWavenumber: System's central wavelength.
%   wavenumberVect: Vector with interrogating wavenumbers.
%   freqXVect, freqYVect: Lateral spatial frequency axes in rad/mm^-1.
%   depthRange: Nyquist-limited axial ranging depth.
%   depthRange6dB: Pulse-limited 6-dB axial ranging depth.
%   axSampling: Axial sampling in mm^-1.
%   latSampling: Lateral sampling in mm^-1.
%   beamWaistDiam: Theoretical exp(-2) beam diameter in mm.
%   plotPSF: Make plots of lateral and axial PSFs.
%
% Outputs:
%   simParms: Struct with parameteres neccesary for running OCT simulation
%
%
% Authors:  Sebastian Ruiz-Lopera {1*}, B. E. Bouma {1}
%           and Nestor Uribe-Patarroyo {1}.
%
% SRL - BEB - NUP:
%  1. Wellman Center for Photomedicine, Harvard Medical School, Massachusetts
%     General Hospital, 40 Blossom Street, Boston, MA, USA.
%
%	* <srulo@mit.edu>
%
% Changelog:
%
% V1.0 (01-29-2024): Initial version released
%
% Copyright Sebastian Ruiz Lopera (2024)
%

PSF_THRESHOLD_LAT = exp(-2);
PSF_THRESHOLD_AX = 0.5;

if nargin == 0
  error('At the least one input is required');
else
  % If options are provided, unpack them
  StructToVars(inputParms);
end

if ~exist('jDGDDretRange', 'var') || isempty(jDGDDretRange)
  jDGDDretRange = [0 0; 0 0];
end
if ~exist('jDGDDangle', 'var') || isempty(jDGDDangle)
  jDGDDangle = [0 0; 0 0];
end

% Make variables gpuArray is using GPU
if useGPU
  varType = {'single', 'gpuArray'};
  ThisLinspace = @gpuArray.linspace;
  ToSingle = @(x) gpuArray(single(x));
else
  varType = {'single'};
  ThisLinspace = @linspace;
  ToSingle = @single;
end

% Get expected tomogram size
nZ = tomSize(1);
nX = tomSize(2);
nY = tomSize(3);

% Central wavenumber
centralWavenumber = 2 * pi * centralWavelength /...
  (centralWavelength ^ 2 - (wavelengthRange / 2) ^ 2);
% Wavenumber range
wavenumberRange = 2 * pi * (wavelengthRange / ...
  (centralWavelength ^ 2 - (wavelengthRange / 2) ^ 2));
% Wavenumber limits
wavenumberLims = centralWavenumber + wavenumberRange * [-1 1] / 2;
% Vector of wavenumber
wavenumberVect = ThisLinspace(wavenumberLims(1), wavenumberLims(2), nK)';
% Wavenumber sampling
if ~exist('deltaWavenumber', 'var') || isempty(deltaWavenumber)
  deltaWavenumber = wavenumberRange / nK;
end
% One-sided axial scan range
depthRange = pi * nK / wavenumberRange / 2;
% One-side 6-dB axial scan range
depthRange6dB = 2 * log(2) / deltaWavenumber;
% Axial sampling
axSampling = 2 * depthRange / nZ;
% Hanning window FWHM
hanningPSF = abs(fftshift(fft(fftshift(padarray(hanning(nK),...
  nK * (64 - 1) / 2), 1), [], 1), 1)) .^ 2;
HanningFWHMum = (find(hanningPSF >= 0.5 * max(hanningPSF), 1, 'last') -...
  find(hanningPSF >= 0.5 * max(hanningPSF), 1, 'first')) * axSampling * 1e6 / 64;
% Hanning FWHM resolution
HanningAxResUm = HanningFWHMum / 2 / (4 / 3); % Divide by 2 (dobule pass) and index of refraction 4/ 3
% Guassian FWHM resolution
GaussianAxResUm = 2 * log(2) / pi * centralWavelength ^ 2 / wavelengthRange / (4 / 3);% Divide by index of refraction 4/3

% Confocal constant
alphaParm = 2 / numAper;
% exp(-2) beam intensity waist diameter
beamWaistDiam = 2 * alphaParm / centralWavenumber;
% Lateral sampling, considering sampling factor
latSampling = beamWaistDiam / 2 / xNyquistOversampling;
% Confocal parameter
confocalParm = alphaParm ^ 2 / centralWavenumber * 1e6 / 2;

% One-sided lateral scan range
xSize = latSampling * nX / 2;
ySize = latSampling * nY / 2;
% Cartesian coordinate
zVect = single(ThisLinspace(0, depthRange, nZ));
xVect = single(ThisLinspace(-xSize, xSize - latSampling, nX));
yVect = ThisLinspace(- ySize, ySize - latSampling, nY); single(0);

% Sensitivity fall-off
zVectNyq = single(ThisLinspace(-depthRange, depthRange - 2 * depthRange / nK, nK));
sensitivity = exp(-4 * log(2) * zVectNyq .^ 2 / depthRange6dB ^ 2);

% Frequency coordinates
freqXVect = single(ThisLinspace(-0.5, 0.5 - 1 / (2 * nX), 2 * nX)) /...
  (latSampling / 2) * 2 * pi; %
freqYVect(1,1,1,:) = single(ThisLinspace(-0.5, 0.5 - 1 / (2 * nY), 2 * nY)) /...
  (latSampling / 2) * 2 * pi;

% To get axial and lateral PSFs and resolutions simulate a single scatterer
% Oversample in both directions
latSamplingOS = latSampling / 256 * xNyquistOversampling;
nXOS = 1024;
nZOS = 64 * nZ;
axSamplingOS = 2 * depthRange / nZOS;
% New vector of positions
xVectOS = single(ThisLinspace(-latSamplingOS * nXOS / 2,...
  latSamplingOS * nXOS / 2 - latSamplingOS, nXOS));
zVectOS = single(ThisLinspace(-depthRange, depthRange, nZOS));
% Frequency coordinates
freqXVectOS = single(ThisLinspace(-0.5, 0.5 - 1 / (2 * nXOS), 2 * nXOS)) /...
  (latSamplingOS / 2) * 2 * pi; %
% Single scatterer
% Initialize fringes
fringesLatPSF = zeros(nK, nXOS, varType{:});
% Raster scan sample
for thisXScan = 1:nXOS
  fringesLatPSF(:, thisXScan, 1) =  ForwardModel_PointScatterers_FreqLowNA_3D(...
    ones(1, 1, 1), zeros(1, 1, 1), -xVectOS(thisXScan), zeros(1, 1, 1),...
    wavenumberVect, centralWavenumber,...
    freqXVectOS, 1, alphaParm, 0, 0, 1, false, true, 1);
end
fringesPSF = fringesLatPSF .* 1i ./ ((2 * pi) .^ 2) .* hanning(nK) ./ wavenumberVect;
tomPSF = fftshift(fft(fftshift(padarray(fringesPSF, (nZOS - nK) / 2, 'both'), 1), [], 1), 1);
% Lateral PSF
latPSF = sum(abs(tomPSF(end/2 , :, 1, :)) .^ 2, 4);
normLatPSF = latPSF / max(latPSF);
% Find exp(-2) diamater
[~, w0px] = max(normLatPSF >= PSF_THRESHOLD_LAT);
latPSFDiamUm = ((nXOS - (2 * w0px)) * latSamplingOS) * 1e6;
% Axial PSF
axPSF = abs(tomPSF(:, end / 2)) .^ 2;
normAxPSF = axPSF / max(axPSF);
% Find exp(-2) diamater
[~, deltaZpx] = max(normAxPSF > PSF_THRESHOLD_AX);
axPSFDiamUm = ((nZOS - 2 * deltaZpx) * axSamplingOS) * 1e6;
% Resolution volume
resolutionVol = axPSFDiamUm * 1e-6 * beamWaistDiam ^ 2;
% Sampling volume
samplingVol = axSampling * latSampling ^ 2;

if plotPSF
  % Options for figures
  set(0, 'defaultTextInterpreter', 'LaTex')
  set(0, 'defaultAxesTickLabelInterpreter', 'LaTex')
  set(groot,'defaultLegendInterpreter','latex');
  set(0, 'DefaultAxesFontSize', 20)
  % Plot lateral psf
  figure(plotPSF), plt1 = plot(xVectOS * 1e6, normLatPSF, 'r', 'linewidth', 2); hold on
  plot(latPSFDiamUm * [-1 1] / 2, PSF_THRESHOLD_LAT * [1 1], 'r--', 'linewidth', 2)
  % Plot axial psf
  plt2 = plot(zVectOS' * 1e6, normAxPSF, 'b', 'linewidth', 2); grid on, grid minor
  xlabel('$\mu$m'), ylabel('Normalized PSF')
  % Add psf widths
  xlim(1.5 * max(latPSFDiamUm, axPSFDiamUm) * [-1 1])
  plot(axPSFDiamUm * [-1 1] / 2, PSF_THRESHOLD_AX * [1 1], 'b--', 'linewidth', 2), hold off
  % and legend
  legend([plt1, plt2], {sprintf('Lateral axis %.2f %s', latPSFDiamUm, '$\mu$m'),...
    sprintf('Axial axis %.2f %s', axPSFDiamUm, '$\mu$m')}, 'fontsize', 20, 'Interpreter','latex')
end

% Generate system's Jones matrices to emulate DGDD
% We are going to simulate DGDD as a single linear retarder with a
% birefringence linearly increasing with wavenumber
jADn = linspace(single(jDGDDretRange(1, 1)), jDGDDretRange(1, 2), nK);
jBDn = linspace(single(jDGDDretRange(2, 1)), jDGDDretRange(2, 2), nK);
% and a given optic axis
jArotM = [cos(jDGDDangle(1)), sin(jDGDDangle(1));...
          -sin(jDGDDangle(1)), cos(jDGDDangle(1))];
jBrotM = [cos(jDGDDangle(2)), sin(jDGDDangle(2));...
          -sin(jDGDDangle(2)), cos(jDGDDangle(2))];
% Compute the system matrices, J_A and J_B
% Initialize matrices. Anti-diagonal elements are zeros
systemMatsJAB = zeros(2, 2, 2, nK, varType{:});
% First elements are ones (fast axis)
systemMatsJAB(1, 1, :, :) = ones(1, 1, 2, nK, varType{:});
% Forth element are given by retardance
systemMatsJAB(2, 2, 1, :) = exp(-1i * jADn);
systemMatsJAB(2, 2, 2, :) = exp(-1i * jBDn);
% and now let's project onto the reference axes
for thisK = 1:nK
  systemMatsJAB(:, :, 1, thisK) = jArotM.' * systemMatsJAB(:, :, 1, thisK) * jArotM;
  systemMatsJAB(:, :, 2, thisK) = jBrotM.' * systemMatsJAB(:, :, 2, thisK) * jBrotM;
end

% Output parameters
simParms = struct( ...
  'centralWavenumber', centralWavenumber,... Central wavenumber in rad/m^-1
  'wavenumberVect', wavenumberVect,... Vector with interrogating wavenumber in rad/m^-1
  'freqXVect', freqXVect, 'freqYVect', freqYVect,... Lateral spatial frequencies plane in rad/m^-1
  'alphaParm', alphaParm,... alpha = 1 = 2 / NA
  'depthRange', depthRange,... Nyquist-limited, one-sided ranging depth
  'depthRange6dB', depthRange6dB,... Pulse-limited, one-sided 6-dB ranging depth
  'axSampling', axSampling, 'latSampling', latSampling,... Axial and lateral sampling in m
  'beamWaistDiamUm', beamWaistDiam,... exp(-2) beam waist diameter in m
  'confocalParm', confocalParm,... Confocal parameter (2x Rayleigh range) in um
  'xSize', xSize, 'ySize', ySize,... Lateral plane size in m
  'zVect', zVect, 'zVectNyq', zVectNyq, 'xVect', xVect, 'yVect', yVect,... Vector with interrogated axial and lateral axes
  'sensitivity', sensitivity,... Sensitivity fall-off
  'axPSFDiamUm', axPSFDiamUm, 'latPSFDiamUm', latPSFDiamUm,... exp(-2) diameter of axial and lateral PSFs
  'resolutionVol', resolutionVol, 'samplingVol', samplingVol,... Resolution and sampling volumes in um^-3
  'systemMatJAB', systemMatsJAB); % System matrices J_A and J_B

% Retrieved from GPU
fieldNames = fieldnames(simParms);
for thisField = 1:numel(fieldNames)
  simParms.(fieldNames{thisField}) = gather(simParms.(fieldNames{thisField}));
end

end
