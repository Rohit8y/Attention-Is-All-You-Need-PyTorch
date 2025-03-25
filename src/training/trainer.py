import torch
import wandb
from tqdm import tqdm


def train_block(model, optimizer, scheduler, tgt_tokenizer, train_dataloader, epoch, device, loss_fn, args):
    torch.cuda.empty_cache()
    model.train()
    batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
    iter_step = -1
    for batch in batch_iterator:
        src = batch['encoder_input'].to(device)  # (b, seq_len)
        tgt = batch['decoder_input'].to(device)  # (B, seq_len)
        src_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
        tgt_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)

        # Feed into the model
        proj_output = model(src, src_mask, tgt, tgt_mask)
        # Compare the output with the label
        label = batch['target_output'].to(device)  # (B, seq_len)

        # Compute the loss using a simple cross entropy
        loss = loss_fn(proj_output.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1))
        batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

        # Backpropagate the loss
        loss.backward()

        # Update the weights
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        iter_step += 1

        # Log metrics
        if not args.skip_wandb:
            wandb.log({"epoch": epoch, "step": iter_step, "loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

        iter_step += 1
