import torch
import torch.nn as nn
import os
import os.path as osp
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import numpy as np
import open3d as o3d
import sys
sys.path.insert(0, osp.join('$ROOTDIR/''python'))
import handler_subject_data as subject_handler
import handler_calib as calib_handler
import handler as capture_handler
import constants
import utils_3d
import trimesh
import psutil
import subprocess
import csv
def run_net(args, config, train_writer=None, val_writer=None):

    start_time = time.time()

    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
   
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)


  
    
    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # print model info
    print_log('Trainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    print_log('Untrainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch-1)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
 
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        # n_batches = 8
        
       
        for idx, (taxonomy_ids, model_ids, obj_ID_used, data_input, data_output, data_normals) in enumerate(train_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]
            obj_ID_used = obj_ID_used[0]
            sample_min_losses = []
            # obj_ID_used = obj_ID_used[sample_idx]
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            # print("n_batches: ", n_batches)
            
            
            if  'PCN' in dataset_name or dataset_name == 'Completion3D' or "HOHpc" in dataset_name: #
                
                input_object = data_input.cuda() if data_input is not None else None

                output_object = data_output.cuda() if data_output is not None else None
                

                if input_object is None or torch.all(input_object == 0):
                    continue
                if output_object is None or torch.all(output_object == 0):
                    continue

                

            elif 'ShapeNet' in dataset_name:
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1

  
            
            ret = base_model(input_object)
            
            sparse_loss, dense_loss = base_model.module.get_loss(ret, output_object, epoch)
         
            _loss = sparse_loss + dense_loss 
            _loss.backward()
           
            


            # forward
            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
                
            else:
                
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
                    
    


            if args.distributed:
                torch.cuda.synchronize()

    
            n_itr = epoch * n_batches + idx 
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)

            if config.scheduler.type == 'GradualWarmup':
                if n_itr < config.scheduler.kwargs_2.total_epoch:
                        scheduler.step()
            
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
           
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

         
        losses_list = ['%.4f' % l for l in losses.val()]
           

        # Assuming you have a CSV file path
        csv_file_path = f"GraspPC/losses/{name}.csv"
        
        # Check if the file exists, create it if not
        if not os.path.isfile(csv_file_path):
            with open(csv_file_path, 'w', newline='') as csvfile:
                # Writing a spaced header with epoch and column names
                header = ['Epoch'] + [f'Loss_{i}' for i in range(1, len(losses_list) + 1)]
                csv_writer = csv.writer(csvfile, delimiter=' ')
                csv_writer.writerow(header)

        # Writing epoch and losses.val to CSV file
        with open(csv_file_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=' ')
            # Writing the epoch and losses for each epoch
            csv_writer.writerow([epoch] + losses_list)
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)   
        if(epoch % 50 == 0 ):
            builder.save_checkpoint(base_model, optimizer, epoch, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     

        
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()
    end_time = time.time()

    training_duration = end_time - start_time
    print("Training time: {:.2f} seconds".format(training_duration))



def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):
    start = time.time()
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1
    
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, obj_ID_used, data_input, data_output, data_normals) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]
            obj_ID_used = obj_ID_used[0]
            sample_metrics = {}
            

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if  'PCN' in dataset_name or 'ProjectShapeNet'or "HOHpc"   in dataset_name: 
                input_object = data_input.cuda() if data_input is not None else None
                output_object  = data_output.cuda() if data_output is not None else None
                
                if input_object is None or torch.all(input_object == 0):
                    continue
                if output_object is None or torch.all(output_object == 0):
                    continue
                
                ret = base_model(input_object)
                coarse_points1 = ret[0]
                output1_dense = ret[-1]
                
                
                #save input ptclds
                input_object_ptcld = input_object[0].cpu().numpy()  # Assuming batch size is 1
                input_object_ptcld = trimesh.PointCloud(input_object_ptcld)
                
                output_file = f"GraspPC/outputs/{name}/input_{taxonomy_id}_{model_id}_{obj_ID_used}.ply"
                input_object_ptcld.export(output_file)

                
                output1_dense_np = output1_dense[0].cpu().numpy()  # Assuming batch size is 1

                output1_file_path = f"GraspPC/outputs/{name}/output1_pred_{taxonomy_id}_{model_id}_{obj_ID_used}.ply"

                output1_dense_ptcld = trimesh.PointCloud(output1_dense_np)
 
                output1_dense_ptcld.export(output1_file_path)
                
                
    return 
