##################################  Unzip aedat files  ##################################
1. Download the data and put them to 'data' file
2. Chose the data name and saving name; 
3. run rawdata2matlab(inputname,outputname);
(e.g. rawdata2matlab('/data/05_Event/RealData/aedat_data/camerashake1.aedat','/data/05_Event/RealData/aedat_data_imgs/camerashake1/');)
################################## Generate event bins ##################################
4. run generate_eventbin(input, videoname, startframe, endframe)
(e.g. generate_eventbin('/data/05_Event/RealData/aedat_data_imgs/public/rotatevideonew2_6/', '/data/05_Event/RealData/aedat_data_imgs/public/rotatevideonew2_6_dvs/', 2, 55);)