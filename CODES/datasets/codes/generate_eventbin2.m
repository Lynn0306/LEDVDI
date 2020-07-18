% save the event bin into 1 file
function [] = generate_eventbin2(input, output, startframe, endframe)
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
    num_eventbins = 18;  % M=3, N=3.
    %% output dir
    imgs_path = [output 'blurred/'];
    if ~exist(imgs_path, 'dir'), mkdir(imgs_path); end
    %% prepare data
    y_o = double(matlabdata.data.polarity.y);
    x_o = double(matlabdata.data.polarity.x);
    pol_o = double(matlabdata.data.polarity.polarity);
    pol_o(pol_o==0) = -1;
    t_o = double(matlabdata.data.polarity.timeStamp) ./ timescale;
    %%
    event_bins = zeros(num_eventbins * (endframe-startframe+1), h, w);
    for frame = startframe:endframe
        % choose frame and time tag
        t_for = double(matlabdata.data.frame.timeStampStart(frame+1))./ timescale - double(matlabdata.data.frame.timeStampEnd(frame))./ timescale;
        t_back = double(matlabdata.data.frame.timeStampStart(frame))./ timescale - double(matlabdata.data.frame.timeStampEnd(frame-1))./ timescale;
        eventstart = double(matlabdata.data.frame.timeStampStart(frame))./ timescale + t_shift - t_back/2;
        eventend = double(matlabdata.data.frame.timeStampEnd(frame))./ timescale + t_shift + t_for/2;
        exptime = eventend - eventstart;
        exptime = exptime / num_eventbins;
        for i = 1:num_eventbins
            %% divide the events through time. neighbor interval.
            idx = (t_o>=(eventstart+exptime*(i-1)))&(t_o<(eventstart+exptime*i));
            x = x_o; y = y_o; pol = pol_o;
            y(idx~=1)=[];
            x(idx~=1)=[];
            pol(idx~=1)=[];
            % generate event bins
            for event_id = 1:length(x)
                event_bins(i+(frame-startframe)*num_eventbins, y(event_id,1), x(event_id,1))= event_bins(i+(frame-startframe)*num_eventbins, y(event_id,1), x(event_id,1)) + pol(event_id,1);
            end
        end
        %% save blurry frames
        blur = matlabdata.data.frame.samples{frame};
        blur = mat2gray(blur);
        imwrite(uint8(blur*255), fullfile(imgs_path, [num2str(frame,'%04d') '.png']));
    end
    save(fullfile([output, 'EventBin3.mat']), 'event_bins');
