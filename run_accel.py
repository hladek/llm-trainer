import os
import torch

from torch.utils.data import DataLoader
import datasets
import transformers
import argparse
import itertools
import json
import time
import math
import datetime
import shutil

import accelerate

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from torch.optim.lr_scheduler import LRScheduler

# TODO debug if
class TrapezoidLRScheduler(LRScheduler):
    def __init__(self, optimizer, base_lr, warmup_steps=0,  end_decay=None, start_decay=None, last_epoch=-1):
        # TODO end decay can be NONE
        if start_decay is None:
            start_decay = end_decay
        self.warmup_steps = warmup_steps
        self.start_decay = start_decay
        self.end_decay = end_decay
        self.base_lr = base_lr
        if start_decay is not  None:
            assert end_decay is not None
        assert start_decay is None or start_decay > warmup_steps
        assert end_decay is None or end_decay > start_decay
        #assert self.warmup_steps <= self.start_decay <= self.end_decay, "Invalid phase boundaries"
        super(TrapezoidLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1
        scale = 1.0

        if current_step < self.warmup_steps:
            scale = current_step / self.warmup_steps
        elif self.start_decay is None or current_step < self.start_decay:
            scale = 1.0
        elif current_step <= self.end_decay:
            scale = (self.end_decay - current_step) / (self.end_decay - self.start_decay)
        else:
            scale = 0.0

        return [self.base_lr * scale for _ in self.optimizer.param_groups]


def is_rank_0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def log_value(value,step):
    if is_rank_0():
      print(value)
    #wandb.log(value,step=step)

def log_info(msg):
    if is_rank_0():
        print(msg)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a transformers model on a Masked Language Modeling task")
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
       help="The path of the dataset name of the dataset to use (via the datasets library).",

    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
       help="The path of the model weights to load).",

    )
    parser.add_argument(
       "--scheduler",
        type=str,
        default=None,
        help="Scheduler  type. Possible are adafactor, cosine and constant.",
        choices=["cosine","constant","adafactor"],
    )
    parser.add_argument(
        "--unfreeze",
        type=str,
        default=None,
       help="Freeze all layers except those given, separated by comma",

    )
    parser.add_argument(
        "--resume_step",
        type=int,
        default=None,
        help="Step to resume",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
       help="The path to the model config. ",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
       help="The path to the tokenizer ",
    )
    parser.add_argument(
        "--model_template_path",
        type=str,
        default=None,
       help="The path to the model template. the template shoud containe the model configuration  and weights. ",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=512,
        help="Evaluation interval in steps Default 512. One step is device batch_size x sequece_lengt tokens",
    )
    
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        help="Per device batch size. Must be set ",
    )

    parser.add_argument(
        "--checkpoint_steps",
        type=int,
        default=1024,
        help="Checkpoint interval in steps Default 1024",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Logging interval in steps Default 100",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=None,
        help="Maximum training steps. One step is per_device_batch_size * num_processes sequences",
    )
    parser.add_argument(
        "--lr_decay_start",
        type=int,
        default=None,
        help="Starting training step of the linear decay of the learning rate.  ",
    )
    parser.add_argument(
        "--train_hours",
        type=int,
        default=None,
        help="Maximum training time in hours. ",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--collator",
        type=str,
        default="mlm",
        help="Collator type. Possible are mlm and t5.",
        choices=["mlm","t5"],
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="Optimizer type. Possible are adamw (torch) and sgdsai (custom)",
        choices=["adamw","adamw8","sgdsai","adafactor","muon"],
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay, optimizer regularization",
    )
    parser.add_argument(
        "--gradient_clipping",
        type=float,
        default=1.0,
        help="Gradient clipping",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Probability of masked token. Default 0.15",
    )
    parser.add_argument(
        "--t5_input_size",
        type=int,
        default=512,
        help="Input size for T5 model. Output size and dataset input size is calculated fro this  value",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=51000,
        help="Warmup for learning. In steps.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4, 
        help="Learning rate"
    )
    parser.add_argument(
        "--min_learning_rate",
        type=float,
        default=1e-6, 
        help="Minimal learning rate. Applicable with cosine scheduler"
    )
    parser.add_argument(
        "--mean_noise_span_length",
        type=float,
        default=3.0, 
        help="Mean span length of masked tokens for T5."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Enbale WandB and set the project.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Tags of the run, split with comma",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Notes  of the run",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    return parser.parse_args()

def evaluate_model(accelerator,model,eval_dataloader,log_dir,batch_size,step,collator_name):
    log_info("evaluating")
    model.eval()
    elosses = []

    with torch.no_grad():
        for estep, ebatch in enumerate(eval_dataloader):
            outputs = model(**ebatch)

            eloss = outputs.loss
            elosses.append(accelerator.gather_for_metrics(eloss.repeat(batch_size)))
    model.train()

    elosses = torch.cat(elosses)
    eval_loss = torch.mean(elosses)

    weights_l2 = sum(p.detach().norm(2).item() ** 2 for p in model.parameters()) ** 0.5

    accelerator.log(
        {
            "train/eval_loss": eval_loss,
            "train/weights_l2":weights_l2
        },step=step
    )
    elosses = []
    step_start_time = time.time()
    eval_loss = 0
    eval_steps = 0
    log_info("done evaluating " + str(time.time() - step_start_time ))

def save_model(accelerator,model,output_dir,step):
    log_info("Saving model at step " + str(step))
    check_dir = output_dir + "/step_{}".format(step)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        check_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    accelerator.save_state(output_dir + "/last_state")

def get_muon_optimizer( model, learning_rate=1e-3, weight_decay=0.1):
    from moonlight import Muon
    muon_params = [
        p
        for name, p in model.named_parameters()
        if p.ndim >= 2 and "embed" not in name and "head" not in name
    ]
    adamw_params = [
        p
        for name, p in model.named_parameters()
        if not (
            p.ndim >= 2 and "embed" not in name and "head" not in name
        )
    ]

    return Muon(
        lr=learning_rate,
        wd=weight_decay,
        muon_params=muon_params,
        adamw_params=adamw_params,
    )

def train():
    args = parse_args()
    d= datetime.datetime.now()
    output_dir = args.output_dir + d.strftime("/%y%m%d_%H%M")
    log_info(output_dir)
    log_with = None
    if args.wandb_project:
        log_with = "wandb"
    if args.scheduler is None:
        assert args.optimizer == "adafactor", "Please set scheduler"

    from accelerate import DistributedDataParallelKwargs
    kwargs_handlers = None
    if args.unfreeze is not None:
        kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation, log_with=log_with,project_dir=output_dir,mixed_precision=args.mixed_precision,kwargs_handlers=kwargs_handlers)
    accelerate.utils.set_seed(42)
    # Step 2: Initialized monitor with DeepSpeed config (get DeepSpeed config object, if needed)
    # defines LOCAL_RANK
    rank = int(os.environ.get("RANK", "0")) 
    # Load Dataset
    p = args.dataset_path
    if not os.path.exists(p):
       raise Exception("Path with train data does not exists")

    tokenizer_path = args.tokenizer_path
    if tokenizer_path is None:
        tokenizer_path = args.model_template_path
    if tokenizer_path is None:
       raise Exception("Tokenizer path not set. Use --tokenizer_path or --model_template_path")

    # Load Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path
    )
    train_dataset = None
    eval_dataset = None
    if args.debug:
        # load just small data for debug
        train_dataset = datasets.load_from_disk(args.dataset_path + "/validation")
    else:
        train_dataset = datasets.load_from_disk(args.dataset_path + "/train")

    eval_dataset = datasets.load_from_disk(args.dataset_path + "/test")
    max_batches = len(train_dataset)
    # One device takes per_device_batch_size batches
    # one batch has sequence_length tokens
    db_max_train_steps = int(max_batches / (args.per_device_batch_size * accelerator.num_processes))
    log_info("Max samples (batches) : {}".format(max_batches))
    log_info("Max steps from database per device : {}".format(db_max_train_steps))
    max_train_steps = db_max_train_steps
    if args.train_steps is not None:
        max_train_steps = args.train_steps
    log_info("Max train steps: {}".format(max_train_steps))

    config_path = args.model_config
    if config_path is None:
        config_path = args.model_template_path
    if config_path is None:
        raise Exception("Configuration not found. Use --model_config or --model_template_path")

    # config is mandatory
    model_config = transformers.AutoConfig.from_pretrained(config_path)
    log_info("Creating model")
    # if there are no weights, initialize empty model
    if args.collator == "mlm":
        if args.model_template_path is None:
            # TODO attention implementation as commandline arg
            model = transformers.AutoModelForMaskedLM.from_config(model_config,  attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
        else:
            model = transformers.AutoModelForMaskedLM.from_pretrained(args.model_template_path,config=model_config,  attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    elif args.collator == "t5":
        if args.model_template_path is None:
            model = transformers.T5ForConditionalGeneration._from_config(model_config)
        else:
            model = transformers.T5ForConditionalGeneration.from_pretrained(args.model_template_path,config=model_config)
    assert model is not None
    log_info("Loaded model")
    # Initialise your wandb run, passing wandb parameters and any config information
    if log_with is not None:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config={
	      "args":vars(args),
              "model":vars(model.config)
            },
 #       init_kwargs={"wandb": {"entity": "my-wandb-team"}}
        )
    #torch.compile(model)
    #model.gradient_checkpointing_enable()
    # Data Collator


    data_collator = None
    if args.collator == "mlm":
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    elif args.collator == "t5":
        from t5_data_collator import  DataCollatorForT5MLM, compute_t5_input_and_target_lengths
        #input_length = 922
        #target_length = 206
        input_length = args.t5_input_size
        expanded_inputs_length, target_length = compute_t5_input_and_target_lengths(
            inputs_length=input_length,
            noise_density=args.mlm_probability,
            mean_noise_span_length=args.mean_noise_span_length,
        )
        assert expanded_inputs_length == len(train_dataset[0]["input_ids"]),\
            f"""
            You have specified that the T5 input length should be {args.t5_input_size}.
            In order to do this, the examples in the dataset need to be {expanded_inputs_length} before masking.
            But the examples in the dataset actually appear to be {len(train_dataset[0]['input_ids'])} tokens long.
            """
        data_collator = DataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=args.mlm_probability,
            mean_noise_span_length=args.mean_noise_span_length,
            input_length=input_length,
            target_length=target_length,
            pad_token_id=model.config.pad_token_id,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )


    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, collate_fn=data_collator, batch_size=args.per_device_batch_size,shuffle=True,pin_memory=True
    )
    

    eval_dataloader = DataLoader(
        eval_dataset,collate_fn=data_collator, batch_size=args.per_device_batch_size
    )
    

    log_info(args)
    log_info("\n".join([k for k,v in model.named_parameters()]))
    # for one bit lamb see https://github.com/microsoft/DeepSpeedExamples/blob/master/training/bing_bert/deepspeed_train.py#L401
    # Optimizer grouped parameters
    # som
    # Layer norm and bias have to be excluded from weight decay
    # This is model depentend, check for any new model !
    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln","attn_norm","mlp_norm"]

    optimizer = None
    optimizer_grouped_parameters = [
      {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
        "requires_grad": True,
      },
      {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "requires_grad": True,
      },
    ]
    assert len(optimizer_grouped_parameters[1]["params"]) > 0, "Norms and bias should be excluded from weight decay"
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if args.optimizer == "muon":
        optimizer = get_muon_optimizer(model,learning_rate=args.learning_rate,weight_decay=args.weight_decay)
    elif args.optimizer == "adamw8":
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(optimizer_grouped_parameters,lr=args.learning_rate,betas=(0.9,0.95))
    elif args.optimizer == "adafactor":
        # https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules
        # https://discuss.huggingface.co/t/t5-finetuning-tips/684/3
        if args.scheduler is not None:
            # Adafactor does not have learning rate.
            # this is hack to use constant scheduler
            lr = args.learning_rate
            rs = False
            if args.scheduler == "adafactor":
                lr = None
                rs = True
            optimizer = transformers.optimization.Adafactor(optimizer_grouped_parameters, lr=lr,relative_step=rs,clip_threshold=args.gradient_clipping,scale_parameter=True,warmup_init=False)
        else:
            optimizer = transformers.optimization.Adafactor(optimizer_grouped_parameters, lr=args.learning_rate,relative_step=False,clip_threshold=args.gradient_clipping,scale_parameter=True,warmup_init=False)
        # todo use adafactor scheduler?
    elif args.optimizer == "sgdsai":
        import sgd_sai
        optimizer = sgd_sai.SGD_sai(model.parameters(), lr=args.learning_rate, momentum=0.9, eps=1e-08, weight_decay=args.weight_decay)
    
    # scheduler should be none for adafactor
    lr_scheduler = None
    # scheduler step is called together with optimizer step
    # warmup and decay start have to be compensated with gradient_accumulation
    update_frequency = args.gradient_accumulation / accelerator.num_processes
    if args.scheduler == "cosine":
        lr_scheduler = transformers.optimization.get_scheduler("cosine_with_min_lr",optimizer,num_warmup_steps=int(args.num_warmup_steps / update_frequency)  ,num_training_steps=int(max_train_steps/update_frequency)  ,scheduler_specific_kwargs={"min_lr":args.min_learning_rate})
    elif args.scheduler == "constant":
        #lr_scheduler = transformers.optimization.get_scheduler("constant_with_warmup",optimizer,num_warmup_steps=args.num_warmup_steps ,num_training_steps=max_train_steps)
        sd = None
        ed = None
        if args.lr_decay_start is not None:
            sd = int(args.lr_decay_start/update_frequency)
            ed = int(max_train_steps/update_frequency)
        lr_scheduler = TrapezoidLRScheduler(
          optimizer,
          base_lr=args.learning_rate,
          warmup_steps=int(args.num_warmup_steps/ update_frequency),
          end_decay=ed,
          start_decay=sd
        )
    elif args.scheduler == "adafactor":
        lr_scheduler = transformers.optimization.AdafactorSchedule(optimizer)


    # to prevent indexing errors
    model.resize_token_embeddings(len(tokenizer))
    #model.compile()
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    log_info("Accelerator initialized")
    log_dir = output_dir + "/train_logs/"
    log_info("Arguments")
    log_info(vars(args))
    with accelerator.main_process_first():
        # Can add new values to logs of old training
        os.makedirs(output_dir,exist_ok=True)
        os.makedirs(log_dir,exist_ok=True)
        if not os.path.exists(output_dir + "/model_template"):
            shutil.copytree(config_path,output_dir+ "/model_template")
            with open(log_dir + "/args.json","w") as f:
                json.dump(vars(args),f)
    if args.resume_path is not None:
        # path has to be saved with save_state
        log_info("Resuming training")
        # Load also scheduler and optimizer state
        accelerator.load_state(args.resume_path)
    if args.resume_step is not None:
        # We skip the first `n` batches in the dataloader when resuming from a checkpoint
        active_dataloader = accelerator.skip_first_batches(train_dataloader, args.resume_step)
    else:
        active_dataloader = train_dataloader
    # Training
    model.train()
    if args.unfreeze is not None:
        layers = set(args.unfreeze.split(","))
        # Freeze all layers
        for k,param in model.named_parameters():
            param.requires_grad = False
            for n in layers:
                if n in k:
                    print("Unfrozen:")
                    print(k)
                    #print(param)
                    #stdmean = torch.std_mean(param,keepdim=True) 
                    #torch.nn.init.normal_(param,stdmean[0],stdmean[1])
                    mean = param.mean()
                    std = param.std()
                    param.copy_(torch.normal(mean, std, size=param.shape))
                    param.requires_grad = True
                    break
    step_start_time = time.time()
    train_start_time = time.time()
    last_time = 0
    grad_l2 = 0

    # For profiling
    total_load_duration  = 0
    total_forward_duration  = 0
    total_backward_duration  = 0
    total_optimize_duration  = 0
    total_step_duration = 0

    log_info("Starting training loop")
    start_step = 0
    counter = 1
    if args.resume_step is not None:
        start_step = args.resume_step
    for step, batch in enumerate(active_dataloader,start=start_step):
        load_time = time.time()
        #with torch.autograd.set_detect_anomaly(True):
        # Forward pass

        with accelerator.accumulate(model):
            outputs = model(**batch)
            forward_time =  time.time()
            loss = outputs.loss
            # We keep track of the loss at each epoch
            #total_loss += loss.detach().float()
            accelerator.backward(loss)
            backward_time = time.time()
            if accelerator.sync_gradients:
                # toto treba dat prec ak je optimizer bitsandbytes, kvoli percentile grad clipping
                if  args.optimizer != "adafactor" and args.gradient_clipping is not None:
                    grad_l2 = accelerator.clip_grad_norm_(
                         parameters=model.parameters(),
                         max_norm=args.gradient_clipping,
                         norm_type=2,
                    ).item()
                else:
                    # Just calculate the norm
                    # https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/4
                    grad_l2 = 0
                    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
                    for p in parameters:
                        param_norm = p.grad.detach().data.norm(2)
                        grad_l2 += param_norm.item() ** 2
                    grad_l2 = grad_l2 ** 0.5

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()
        optimize_time = time.time()

        if  step % args.logging_steps == 0:
            onebatch = batch["input_ids"].shape[0] *  batch["input_ids"].shape[1] 
            num_tokens = counter * accelerator.num_processes * onebatch
            d = {"train/loss":loss.detach().item(),
                  "train/step":step,
                  "train/grad_l2": grad_l2,
                  "train/tokens": num_tokens
            }
            if lr_scheduler is not None and args.scheduler != "adafactor":
                lrs =  [group["lr"] for group in optimizer.param_groups]
                d["train/lr"] = lrs[0]
            accelerator.log(d,step=step)
            #accelerator.log(d,log_kwargs={"wandb":{"commit":True}})
            log_file = log_dir +"/" +  str(rank) + "-trainlog.json" 
            with open(log_file,"a") as f:
                print(json.dumps(d),file=f)
            if is_rank_0():
                print(json.dumps(d))
        if step % args.eval_steps == args.eval_steps - 1:
            evaluate_model(accelerator,model,eval_dataloader,log_dir,args.per_device_batch_size,step,args.collator)
        if step % args.checkpoint_steps == args.checkpoint_steps -1:
            save_model(accelerator,model,output_dir,step)
        
        # Logging profiling


        load_duration  = load_time -step_start_time 
        forward_duration  = forward_time - load_time
        backward_duration  = backward_time - forward_time
        optimize_duration  = optimize_time - backward_time
        step_duration = optimize_time -step_start_time 
        
        total_load_duration  += load_duration
        total_forward_duration  += forward_duration
        total_backward_duration  += backward_duration
        total_optimize_duration  += optimize_duration
        total_step_duration += step_duration
    
        average_load_duration = total_load_duration / counter
        average_forward_duration = total_forward_duration / counter
        average_backward_duration = total_backward_duration / counter 
        average_optimize_duration = total_optimize_duration / counter
        average_step_duration = total_step_duration / counter
        counter += 1
        
        if args.debug:
            print("rank: {} step: {} load time  {}/{}".format(rank,step,load_duration,average_load_duration))
            print("rank: {} time of forward  pass  {}/{}".format(rank,forward_duration,average_forward_duration))
            print("rank: {} time of backward pass  {}/{}".format(rank,backward_duration,average_backward_duration))
            print("rank: {} time of optimization  {}/{}".format(rank,optimize_duration,average_optimize_duration))
            print("rank: {} total time of one step {}/{}".format(rank,step_duration,average_step_duration))
        if  step % args.logging_steps == 0:
            accelerator.log({
               "profile/load_duration":load_duration,
               "profile/forward_duration":forward_duration,
               "profile/backward_duration":backward_duration,
               "profile/optimize_duration":optimize_duration,
               "profile/step_duration":step_duration,
            })
        # Stop conditions
        if step >= max_train_steps:
            log_info("early stop at {}".format(step))
            break
        if args.train_hours is not None:
            train_hours = int((time.time() - train_start_time) / 3600)
            if train_hours >= args.train_hours:
                log_info("Reached maximal training time")
                break
        step_start_time = time.time()

    average_load_duration = total_load_duration / counter
    average_forward_duration = total_forward_duration / counter
    average_backward_duration = total_backward_duration / counter 
    average_optimize_duration = total_optimize_duration / counter
    average_step_duration = total_step_duration / counter
    accelerator.log({
       "profile_result/avereage_load_duration":average_load_duration,
       "profile_result/avereage_forward_duration":average_forward_duration,
       "profile_result/avereage_backward_duration":average_backward_duration,
       "profile_result/avereage_optimize_duration":average_optimize_duration,
       "profile_result/avereage_step_duration":average_step_duration,
    })
    save_model(accelerator,model,output_dir,str(start_step + counter) + "_final")
    evaluate_model(accelerator,model,eval_dataloader,log_dir,args.per_device_batch_size,start_step + counter + 1,args.collator)
    accelerator.end_training()

if __name__ == "__main__":
    train()
