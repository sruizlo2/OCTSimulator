function [tom, fringes, simParms, tomNoiseFree] = SimulateTomogram(scattAmp, scattPos, inputParms)

% SimulateTomogram Emulates the OCT signal from a given system and sample
% composed of sub-resolution point scatterers
%
% [tom, fringes, simParms, tomNoiseFree] = SimulateTomogram(scattAmp, scattPos, inputParms)
%
% Inputs:
%   scattAmp: Sample's scattering potential
%   scattPos: Z, X, and Y position of point scatterers in m.
%   inputParms: struct with input parameters:
%   centralWavelength: Central wavelength.
%   wavelengthRange: Full spectral bandwidth.
%   numAper: Numerical aperture.
%   xNyquistOversampling: Factor to oversample laterally (1-> Nyquist).
%   noiseFloorDb: Noise level in dB, added to the complex tom.
%   focalPlane: Location of the focal plane wrt to the sample in m.
%   zRef: Location of zero-path delay wrt sample in m.
%   useGPU: Use GPU to evaluate the forward model.
%   maxPointsBatch: Maximum number of scatterers that will be evaluated at
%                   once. Determines GPU usage.
%
% Outputs:
%   tom: Simulated complex OCT tomogram.
%   fringes: Simulated complex OCT fringes.
%   simParms: Parameters of the simulation
%   tomNoiseFree: Noiseless, simulated complex OCT tomogram.
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

if nargin > 3
    error(sprintf('%s\n%s', 'Must provide amplitude and location of point scatterers and input parameters.',...
      '[tom, fringes, simParms, tomNoiseFree] = SimulateTomogram(scattAmp, scattPos, inputParms)'));
else
    % Unpack options
    StructToVars(inputParms);
end

% Default versboity flag. 0 for nothing, 1 for minimal, > 1 for more
if ~exist('verbosity', 'var') || isempty(verbosity)
  verbosity = false;
end

% Default efficient flag
if ~exist('efficient', 'var') || isempty(efficient)
  efficient = true;
end

% Use this to activate (default) or deactivate confocal function
if ~exist('confocalFuncOn', 'var') || isempty(confocalFuncOn)
  confocalFuncOn = 1;
end

% Use this to activate (default) or deactivate sensitivity fall-off
if ~exist('sensitivityOn', 'var') || isempty(sensitivityOn)
  sensitivityOn = 1;
end

% Default input polarization state
if ~exist('inputPolState', 'var') || isempty(inputPolState)
  inputPolState = [];
end

% Default spectral windows (single hanning)
if ~exist('spectralWindows', 'var') || isempty(spectralWindows)
  spectralWindows = hanning(nK);
elseif sum(size(spectralWindows) > 1) > 2
  if size(spectralWindows(:, :), 1) <= size(spectralWindows(:, :), 2)
    error('Input spectral windows do not have the expected 6D shape.')
  end
  if ndims(spectralWindows) < 6
    warning('Spectral windows were shaped to the expected 6D shape.')
    spectralWindows = permute(spectralWindows(:, :), [1 6 3 4 5 2]);
  end
end

% Generate simulation parameters
inputParms.plotPSF = false; % Do not plot PSF
simParms = GetTomogramParameters(inputParms);
StructToVars(simParms);

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

% Replicate zRef is not a matrix
if length(zRef) == 1
  zRef = repmat(zRef, [1, nX, nY]);
end

% Reshape sample
scattAmp = permute(scattAmp, [2 3 1 4 5]);
scattPosZ = permute(scattPos(:, 1), [3 2 1]);
scattPosX = permute(scattPos(:, 2), [3 2 1]);
scattPosY = permute(scattPos(:, 3), [3 2 1]);

% Initialize fringes
nInputPolState = max(1, size(inputPolState, 2)); % Make sure we have at least one state
sampleSignal = zeros(nK, nX, nY, 2, nInputPolState, varType{:});
% Raster scan sample
if verbosity
  fprintf('Simulating tomogram of size [%d, %d, %d]\n', [nK, nX, nY]);
  tic
end
for thisYScan = 1:nY
  if verbosity > 1
    fprintf('Bscan %d: 0%s', thisYScan, '%')
    progress = 0.1;
  end
  for thisXScan = 1:nX
    % Current beam position
    thisBeamPosX = xVect(thisXScan);
    thisBeamPosY = yVect(thisYScan);
    % Fringes at this beam possition is the contribution of the Gaussian
    % beam at the location of the point sources
    sampleSignal(:, thisXScan, thisYScan, :, :) =  ForwardModel_PointScatterersJones_FreqLowNA_3D(...
      scattAmp, scattPosZ, scattPosX - thisBeamPosX, scattPosY - thisBeamPosY,...
      wavenumberVect, centralWavenumber, freqXVect, freqYVect, alphaParm,...
      focalPlane, zRef(1, thisXScan, thisYScan), maxPointsBatch, efficient,...
      confocalFuncOn, mediaIndex, inputPolState, systemMatJAB);
    if verbosity > 1 && thisXScan >= progress * nX
      fprintf(', %d%s', round(progress * 100), '%');
      progress = progress + 0.1;
    end
  end
  if verbosity
    fprintf('. Done! \n')
  end
end
% Sample backscatering signal
sampleSignal = sqrt(hanning(nK)) .* sampleSignal .* 1i ./...
  ((2 * pi) .^ 2) ./ wavenumberVect;
% Reference signal (unused; assume we have 45º polarization and ideal ref)
% refSignal =  ForwardModel_PointScatterersJones_FreqLowNA_3D(...
%   1, zRef - simParms.depthRange, 0, 0,...
%   wavenumberVect, centralWavenumber, freqXVect, freqYVect, alphaParm,...
%   focalPlane, zRef, maxPointsBatch, efficient, confocalFuncOn, mediaIndex, refPolState);
% Interference fringes
fringes = real(sampleSignal);
% Apply sensitivity fall-off in z-space
if ~sensitivityOn; sensitivity = ones(nK, 1); end
fringes = fftshift(ifft(ifftshift(ifftshift(fft(fftshift(fringes,...
  1), [], 1), 1) .* sensitivity(:), 1), [], 1), 1);
% Noiseless tomogram
if nargout > 3
  tomNoiseFree = ifftshift(fft(fftshift(padarray(fringes .* hanning(nK),...
  (2 * nZ - nK) / 2, 'both'), 1), [], 1), 1);
  tomNoiseFree = tomNoiseFree(end/2+1:end, :, :, :, :, :);
end
% Generate real noise for fringes, considering nK
noiseStd = 10 ^ (noiseFloorDb / 20) / sqrt(nK);
noise = noiseStd * randn(nK, nX, nY, 2, nInputPolState);
% Noisy tomogram, add real noise to fringes and apply hamming
tom = ifftshift(fft(fftshift(padarray((fringes + noise) .* spectralWindows,...
  (2 * nZ - nK) / 2, 'both'), 1), [], 1), 1);
% Extract one copy
tom = tom(end/2+1:end, :, :, :, :, :);
if verbosity
  time = toc;
  fprintf('Simulation time: %.2f min\n', time / 60);
end