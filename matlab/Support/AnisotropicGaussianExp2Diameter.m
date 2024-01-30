function h = AnisotropicGaussianExp2Diameter(diameters, exp2DiamX, exp2DiamY)
  
  % This script and its functions follow the coding style that can be
  % sumarized in:
  % * Variables have lower camel case
  % * Functions upper camel case
  % * Constants all upper case
  % * Spaces around operators
  %
  % Authors:  Néstor Uribe-Patarroyo
  %
  % NUP:
  % 1. Wellman Center for Photomedicine, Harvard Medical School, Massachusetts
  % General Hospital, 40 Blossom Street, Boston, MA, USA;
  % <uribepatarroyo.nestor@mgh.harvard.edu>
  
  % MGH Flow Measurement project (v1.0)
  %
  % Changelog:
  %
  % V1.0 (2014-07-03): Initial version released
  
  if (exp2DiamX ~= 0) && (exp2DiamY ~= 0)
    radii   = (diameters-1)/2;
    
    [x, y] = meshgrid(-radii(1):radii(1),-radii(2):radii(2));
    arg   = -(8 .* x .* x / (exp2DiamX ^ 2) + 8 .* y .* y / (exp2DiamY ^ 2));
    h     = exp(arg);
  elseif (exp2DiamX == 0) && (exp2DiamY ~= 0)
    radii   = (diameters-1)/2;
    
    [x, y] = meshgrid(-radii(1):radii(1),-radii(2):radii(2));
    filtX = x == 0;
    arg   = -(8 .* y .* y / (exp2DiamY ^ 2));
    h     = exp(arg);
    h = h .* filtX;
  elseif (exp2DiamX ~= 0) && (exp2DiamY == 0)
    radii   = (diameters-1)/2;
    
    [x, y] = meshgrid(-radii(1):radii(1),-radii(2):radii(2));
    filtY = y == 0;
    arg   = -(8 .* x .* x / (exp2DiamX ^ 2));
    h     = exp(arg);
    h = h .* filtY;
  else
    h = 1;
  end
  h(h<eps*max(h(:))) = 0;
  
  sumh = sum(h(:));
  if sumh ~= 0
    h  = h/sumh;
  end
end
