function [] = generate_eventbin(input, output, startframe, endframe)
    %% load data
    dataname = input;
    load([dataname 'data.mat']);
    %% parameters
    % Data paremeter
    timescale = 1e6;
    t_shift = -0.04;

    blur = matlabdata.data.frame.samples{startframe};
    blur = mat2gray(blur);
    [h, w] = size(blur);
    num_eventbins = 6;
    num_events_per_images = 0.01 * h * w;
    %% output dir
    videoname = output;
    if ~exist(videoname,'dir'), mkdir(videoname); end
    event_path_loss = [output 'event_bins_loss/'];
    if ~exist(event_path_loss, 'dir'), mkdir(event_path_loss); end
    event_path = [output 'event_bins/'];
    if ~exist(event_path, 'dir'), mkdir(event_path); end
    event_path_fixednum = [output 'event_bins_fixednum/'];
    if ~exist(event_path_fixednum, 'dir'), mkdir(event_path_fixednum); end
    imgs_path = [output 'images/'];
    if ~exist(imgs_path, 'dir'), mkdir(imgs_path); end
    %% prepare data
    y_o = double(matlabdata.data.polarity.y);
    x_o = double(matlabdata.data.polarity.x);
    pol_o = double(matlabdata.data.polarity.polarity);
    pol_o(pol_o==0) = -1;
    t_o = double(matlabdata.data.polarity.timeStamp) ./ timescale;
    %%
    for frame = startframe:endframe
        % choose frame and time tag
        t_for = double(matlabdata.data.frame.timeStampStart(frame+1))./ timescale - double(matlabdata.data.frame.timeStampEnd(frame))./ timescale;
        t_back = double(matlabdata.data.frame.timeStampStart(frame))./ timescale - double(matlabdata.data.frame.timeStampEnd(frame-1))./ timescale;
        eventstart = double(matlabdata.data.frame.timeStampStart(frame))./ timescale + t_shift - t_back/2;
        eventend = double(matlabdata.data.frame.timeStampEnd(frame))./ timescale + t_shift + t_for/2;
%         eventstart = double(matlabdata.data.frame.timeStampStart(frame))./ timescale - t_back/2;
%         eventend = double(matlabdata.data.frame.timeStampEnd(frame))./ timescale + t_for/2;
        exptime = eventend - eventstart;
        exptime = exptime / num_eventbins;
        event_bin = zeros(num_eventbins, h, w);
        event_bin_loss = zeros(num_eventbins, h, w);
        event_bin_fixednum = zeros(num_eventbins, h, w);
        for i = 1:num_eventbins
            %% divide the events through time. 1/4 neighbor interval.
            idx = (t_o>=(eventstart+exptime*i-exptime/4))&(t_o<=(eventstart+exptime*i));
            x = x_o; y = y_o; pol = pol_o;
            y(idx~=1)=[];
            x(idx~=1)=[];
            pol(idx~=1)=[];
            % generate event bins
            for event_id = 1:length(x)
                event_bin_loss(i, y(event_id,1), x(event_id,1)) = event_bin_loss(i, y(event_id,1), x(event_id,1)) + pol(event_id,1);
            end
            %% divide the events through time. neighbor interval.
            idx = (t_o>=(eventstart+exptime*(i-1)))&(t_o<(eventstart+exptime*i));
            x = x_o; y = y_o; pol = pol_o;
            y(idx~=1)=[];
            x(idx~=1)=[];
            pol(idx~=1)=[];
            % generate event bins
            for event_id = 1:length(x)
                event_bin(i, y(event_id,1), x(event_id,1)) = event_bin(i, y(event_id,1), x(event_id,1)) + pol(event_id,1);
            end
            %% divide the events through number of events. 0.01 events/pixel
            idx = find(t_o<=(eventstart+exptime*i), num_events_per_images, 'last');
            x = x_o(idx); 
            y = y_o(idx); 
            pol = pol_o(idx);
            % generate event bins
            for event_id = 1:length(x)
                event_bin_fixednum(i, y(event_id,1), x(event_id,1)) = event_bin_fixednum(i, y(event_id,1), x(event_id,1)) + pol(event_id,1);
            end
        end
        save(fullfile(event_path, [num2str(frame,'%04d') '.mat']), 'event_bin');
        save(fullfile(event_path_loss, [num2str(frame,'%04d') '.mat']), 'event_bin_loss');
        save(fullfile(event_path_fixednum, [num2str(frame,'%04d') '.mat']), 'event_bin_fixednum');
        %% devide the event through event number. 
        %% save result
        blur = matlabdata.data.frame.samples{frame};
        blur = mat2gray(blur);
        imwrite(uint8(blur*255), fullfile(imgs_path, [num2str(frame,'%04d') '.png']));
    end