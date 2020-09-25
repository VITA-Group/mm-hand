import time

import torch
from data.mmhand_dataset_data_loader import MMHandDatasetDataLoader

from models.MMHandModel import MMHandModel
from options.train_options import TrainOptions
from util.visualizer import Visualizer

if __name__ == "__main__":

    opt = TrainOptions().parse()
    data_loader = MMHandDatasetDataLoader(opt)
    torch.cuda.set_device(opt.local_rank)
    model = MMHandModel(opt)
    model.pprint('#training images = %d' % len(data_loader))
    model.pprint("model [%s] was created" % (model.name()))
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0 and model.master:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(),
                                                   epoch, save_result)

            if total_steps % opt.print_freq == 0 and model.master:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(
                        epoch,
                        float(epoch_iter) / dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0 and model.master:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if opt.distributed:
            data_loader.set_epoch(epoch)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            if model.master:
                model.save('latest')
                model.save(epoch)

        model.pprint('End of epoch %d / %d \t Time Taken: %d sec' %
                     (epoch, opt.niter + opt.niter_decay,
                      time.time() - epoch_start_time))
        model.update_learning_rate()
