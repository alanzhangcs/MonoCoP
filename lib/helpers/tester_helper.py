import os
import tqdm
import shutil

import torch
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from lib.helpers.benchmark import flop_count
import time
import numpy as np

class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, train_cfg=None, model_name='monocop'):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.train_cfg = train_cfg
        self.model_name = model_name
        self.checkpoint_path = cfg["checkpoint_path"] if "checkpoint_path" in cfg.keys() else None

    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        if self.checkpoint_path is not None:
            checkpoint_path = self.checkpoint_path
        else:
            checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
        assert os.path.exists(checkpoint_path)
        load_checkpoint(model=self.model,
                        optimizer=None,
                        filename=checkpoint_path,
                        map_location=self.device,
                        logger=self.logger)
        self.model.to(self.device)
        # self.calculate_flops()
        self.inference()
        self.evaluate()
        
    
    def calculate_flops(self):


        model = self.model
        model.eval()

        warmup_steps = 5
        measure_steps = 20

        infer_times = []

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Params: {total_params / 1e6:.2f} M")

        with torch.no_grad():
            for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
                if batch_idx >= warmup_steps + measure_steps:
                    break
                inputs = inputs.to(self.device, dtype=torch.float32)
                calibs = calibs.to(self.device, dtype=torch.float32)
                img_sizes = info["img_size"].to(self.device, dtype=torch.long)
        

                if batch_idx == 0:
                    gflops = flop_count(model, (inputs, calibs, targets, img_sizes))
                    print(f"GFLOPs: {sum(gflops.values()):.2f}")


                torch.cuda.synchronize()
                start_time = time.time()
                _ = model(inputs, calibs, targets, img_sizes)
                torch.cuda.synchronize()
                end_time = time.time()

                elapsed = end_time - start_time
                if batch_idx >= warmup_steps:
                    infer_times.append(elapsed)

            # 时间与 FPS 输出
            infer_times = np.array(infer_times)
            mean_time = infer_times.mean()
            fps = 1 / mean_time if mean_time > 0 else float('inf')

            print(f"Average inference time per sample: {mean_time * 1000:.2f} ms")
            print(f"FPS: {fps:.2f}")

    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        model_infer_time = 0
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)

            start_time = time.time()
            ###dn
            outputs = self.model(inputs, calibs, targets, img_sizes, dn_args = 0)
            ###
            end_time = time.time()
            model_infer_time += end_time - start_time

            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs, topk=self.cfg['topk'])

            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg.get('threshold', 0.2))

            results.update(dets)
            progress_bar.update()

        print("inference on {} images by {}/per image".format(
            len(self.dataloader), model_infer_time / len(self.dataloader)))

        progress_bar.close()

        # save the result for evaluation.
        self.logger.info('==> Saving ...')
        self.save_results(results)

    def save_results(self, results):
        output_dir = os.path.join(self.output_dir, 'outputs', 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if self.dataset_type == 'KITTI':
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id),
                                           self.dataloader.dataset.get_sample_token(img_id) + '.txt')

            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()

    def evaluate(self):
        results_dir = os.path.join(self.output_dir, 'outputs', 'data')
        assert os.path.exists(results_dir)
        result = self.dataloader.dataset.eval(results_dir=results_dir, logger=self.logger)
        return result
