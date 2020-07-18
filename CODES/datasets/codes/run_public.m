% parse aedat
rawdata2matlab('../data/rotatevideonew2_6.aedat','../parse/rotatevideonew2_6/');
% generate event bins.
generate_eventbin2('../parse/rotatevideonew2_6/', '../dvs/rotatevideonew2_6/', 2, 55);
