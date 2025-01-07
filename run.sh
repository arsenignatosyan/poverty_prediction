python -m modelling.prithvi.finetune_temporal --imagery_path landsat_data --batch_size 256 --normalize
python -m modelling.prithvi.finetune_temporal --imagery_path landsat_data --freeze 9 --batch_size 256 --normalize
python -m modelling.prithvi.finetune_temporal --imagery_path landsat_data --freeze 12 --batch_size 256 --normalize

python -m modelling.prithvi.finetune_spatial --fold 1 --imagery_path landsat_data --batch_size 256 --normalize
python -m modelling.prithvi.finetune_spatial --fold 2 --imagery_path landsat_data --batch_size 256 --normalize
python -m modelling.prithvi.finetune_spatial --fold 3 --imagery_path landsat_data --batch_size 256 --normalize
python -m modelling.prithvi.finetune_spatial --fold 4 --imagery_path landsat_data --batch_size 256 --normalize
python -m modelling.prithvi.finetune_spatial --fold 5 --imagery_path landsat_data --batch_size 256 --normalize

python -m modelling.prithvi.finetune_spatial --fold 1 --imagery_path landsat_data --freeze 9 --batch_size 256 --normalize
python -m modelling.prithvi.finetune_spatial --fold 2 --imagery_path landsat_data --freeze 9 --batch_size 256 --normalize
python -m modelling.prithvi.finetune_spatial --fold 3 --imagery_path landsat_data --freeze 9 --batch_size 256 --normalize
python -m modelling.prithvi.finetune_spatial --fold 4 --imagery_path landsat_data --freeze 9 --batch_size 256 --normalize
python -m modelling.prithvi.finetune_spatial --fold 5 --imagery_path landsat_data --freeze 9 --batch_size 256 --normalize

python -m modelling.prithvi.finetune_spatial --fold 1 --imagery_path landsat_data --freeze 12 --batch_size 256 --normalize
python -m modelling.prithvi.finetune_spatial --fold 2 --imagery_path landsat_data --freeze 12 --batch_size 256 --normalize
python -m modelling.prithvi.finetune_spatial --fold 3 --imagery_path landsat_data --freeze 12 --batch_size 256 --normalize
python -m modelling.prithvi.finetune_spatial --fold 4 --imagery_path landsat_data --freeze 12 --batch_size 256 --normalize
python -m modelling.prithvi.finetune_spatial --fold 5 --imagery_path landsat_data --freeze 12 --batch_size 256 --normalize
