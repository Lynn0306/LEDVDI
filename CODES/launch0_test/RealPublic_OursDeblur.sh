cd ../

python test.py  \
--name test \
--dataset_mode RealContinuous --event_name  EventBin3 \
--eventbins_between_frames 3 \
--test_batch_size 1  --n_threads 0  \
--model Ours_DeblurOnly \
--load_G  './checkpoints/Ours_DeblurOnly_Smaller/net_epoch_0_id_G.pth' \
--test_data_dir  ./datasets/dvs/ \
--output_dir './OutputReal/Ours_Deblur'