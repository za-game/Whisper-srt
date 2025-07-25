模型下載前置
winget install git-lfs
git lfs install
 
模型：
git clone https://huggingface.co/ctranslate2-4you/whisper-medium-ct2-float16
git clone https://huggingface.co/ctranslate2-4you/whisper-medium-ct2-int8-float32

模型下載至./Whisper_srt/model

使用參考

python mWhisperSub.py --model_dir models/whisper_medium_ct2 --compute_type int8_float16 --gpu 0 --sr 48000 --dtype int16 --device 1 --workers 2 --beam 3 --silence 0.01 --min_infer_gap 0.1 --maxhop 3

--model_dir models/whisper_medium_ct2	#模型路徑
--compute_type int8_float16 		#模型型態
--gpu 0 				#指定GPU
--sr 48000				#裝置取樣
--dtype int16				#裝置取樣精度
--device 1				#音訊裝置
--workers 2				#多線程1為單線程 2雙線程
--beam 3				#推理候選數
--best_of 5				#多次解碼beam>1無效
--silence 0.01				#靜音推理條件
--min_infer_gap 0.1			#最短推理間隔
--maxhop 3				#強制推理時間