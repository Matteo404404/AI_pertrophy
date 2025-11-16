# Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,           # CHANGE: was 0
        pin_memory=True,         # ADD
        prefetch_factor=2,       # ADD
        persistent_workers=True  # ADD
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,           # CHANGE: was 0
        pin_memory=True          # ADD
    )