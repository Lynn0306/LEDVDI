
%% parse aedat
% rawdata2matlab(InputPath_Aedat, OutputPath_ParsedData);
rawdata2matlab('../data/rotatevideonew2_6.aedat','../parse/rotatevideonew2_6/');

%% generate event bins.
% generate_eventbin2(InputPath_ParsedData, OutputPath, StartFrame, EndFrame);
generate_eventbin2('../parse/rotatevideonew2_6/', '../dvs/rotatevideonew2_6/', 2, 55);
