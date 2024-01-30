function [tomStokes] = ComplexTomToStokes(tomComplex, polDim, varargin)
  %ComplexTomToStokes(tomComplex, stokesDim) Calc Stokes from complex Tom
  %   polDim indicates the index corresponding to the Jones polarization states
  
  if nargin > 2 && ~isempty(varargin{1}) && strcmpi(varargin{1}, 'onlyI')
    onlyI = true;
  else
    onlyI = false;
  end
  
  dims = 1:ndims(tomComplex);
  nDims = numel(dims);
  colonOp = repmat({':'}, [nDims, 1]);
  
  colonOpLHS = colonOp;
  colonOpRHS1 = colonOp;
  colonOpRHS2 = colonOp;
  colonOpRHS1{polDim} = 1;
  colonOpRHS2{polDim} = 2;
  if ~onlyI
    % Start with V to preallocate memmory
    colonOpLHS{polDim} = 4;
    tomStokes(colonOpLHS{:}) = -2 * imag(tomComplex(colonOpRHS1{:}) .* conj(tomComplex(colonOpRHS2{:})));
    % Now with U
    colonOpLHS{polDim} = 3;
    tomStokes(colonOpLHS{:}) = 2 * real(tomComplex(colonOpRHS1{:}) .* conj(tomComplex(colonOpRHS2{:})));
    % Now Q
    colonOpLHS{polDim} = 2;
    tomStokes(colonOpLHS{:}) = abs(tomComplex(colonOpRHS1{:})) .^ 2 - abs(tomComplex(colonOpRHS2{:})) .^ 2;
  end
  % And I
  colonOpLHS{polDim} = 1;
  tomStokes(colonOpLHS{:}) = abs(tomComplex(colonOpRHS1{:})) .^ 2 + abs(tomComplex(colonOpRHS2{:})) .^ 2;
  
end

