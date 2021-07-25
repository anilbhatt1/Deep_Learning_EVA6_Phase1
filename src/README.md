- Contains py files required for model training, testing & inferencing.
- Content structure is as below:

- |── Alb_transforms.py

- |   ├── Alb_trans  -> Class

|   	├── __init__() -> fn

|   	├── __call__() -> fn

├── main.py  

|   ├── run_main()	-> fn

|   ├── s9_run_main() -> fn

|   ├── s10_run_main() -> fn

├── models.py 

|   ├── CNNNorm -> Class

|   	├── __init__() -> fn

|   	├── __forward__() -> fn

|   ├── CNNBlocks -> Class

|   	├── __init__() -> fn

|   	├── __forward__() -> fn

|   ├── S6_CNNModel -> Class

|   	├── __init__() -> fn

|   	├── __forward__() -> fn

|   ├── S7_CNNModel -> Class

|   	├── __init__() -> fn

|   	├── __forward__() -> fn

|   ├── S7_CNNModel_mixed -> Class

|   	├── __init__() -> fn

|   	├── __forward__() -> fn

|   ├── BasicBlock -> Class

|   	├── __init__() -> fn

|   	├── __forward__() -> fn

|   ├── ResNet -> Class

|   	├── __init__() -> fn

|       ├── _make_layer() -> fn

|   	├── __forward__() -> fn

|   ├── ResNet18() -> fn

|   ├── BasicBlock_Custom -> Class

|   	├── __init__() -> fn

|   	├── __forward__() -> fn

|   ├── ResNet_Custom -> Class

|   	├── __init__() -> fn

|       ├── _make_layer() -> fn

|   	├── __forward__() -> fn

|   ├── ResNet_C() -> fn

|   ├── BasicBlock_Tiny -> Class

|   	├── __init__() -> fn

|   	├── __forward__() -> fn

|   ├── ResNet18_Tiny -> Class

|   	├── __init__() -> fn

|       ├── _make_layer() -> fn

|   	├── __forward__() -> fn

|   ├── ResNet18_TinyImageNet() -> fn

├── s7_model.py

|   ├── S7_CNNModel -> Class

|   	├── __init__() -> fn

|   	├── __forward__() -> fn

|   ├── S7_CNNModel_mixed -> Class

|   	├── __init__() -> fn

|   	├── __forward__() -> fn

├── s8_resnet_cifar10_model.py 

|   ├── BasicBlock -> Class

|   	├── __init__() -> fn

|   	├── __forward__() -> fn

|   ├── ResNet -> Class

|   	├── __init__() -> fn

|       ├── _make_layer() -> fn

|   	├── __forward__() -> fn

|   ├── ResNet18() -> fn

├── s9_resnet_custom.py 

|   ├── BasicBlock_Custom -> Class

|   	├── __init__() -> fn

|   	├── __forward__() -> fn

|   ├── ResNet_Custom -> Class

|   	├── __init__() -> fn

|       ├── _make_layer() -> fn

|   	├── __forward__() -> fn

|   ├── ResNet_C() -> fn

├── test_loss.py

|   ├── test_losses -> Class

|   	├── __init__() -> fn

|   	├── s6_test() -> fn

|   	├── s7_test() -> fn

|   	├── s8_test() -> fn

|   	├── s9_test() -> fn

|   	├── s10_test() -> fn

├── train_loss.py

|   ├── train_losses -> Class

|   	├── __init__() -> fn

|   	├── s6_train() -> fn

|   	├── s7_train() -> fn

|   	├── s8_train() -> fn

|   	├── s9_train() -> fn

|   	├── s10_train() -> fn

├── utilities.py 

|   ├── Alb_trans -> Class

|   	├── __call__() -> fn

|   ├── stats_collector -> Class

|   	├── __init__() -> fn

|   	├── append_loss() -> fn

|   	├── append_acc() -> fn

|   	├── append_img() -> fn

|   	├── append_pred() -> fn

|   	├── append_label() -> fn

|   ├── unnorm_img -> Class

|   	├── __init__() -> fn

|   	├── unnorm_rgb() -> fn

|   	├── unnorm_gray() -> fn

|   	├── unnorm_albumented() -> fn

|   ├── cifar10_plots -> Class

|   	├── __init__() -> fn

|   	├── unnormalize_cifar10() -> fn

|   	├── plot_cifar10_train_imgs() -> fn

|   	├── plot_cifar10_gradcam_imgs() -> fn

|   	├── plot_cifar10_misclassified() -> fn

|   ├── tiny_imagenet_plots -> Class

|   	├── __init__() -> fn

|   	├── unnormalize_np_tensor() -> fn

|   	├── show_train_images() -> fn

|   	├── plot_tinyimagenet_misclassified() -> fn

|   	├── plot_cifar10_misclassified() -> fn

|   ├── ctr() -> fn

|   ├── CIFAR10_data_prep -> fn

|   ├── S9_CIFAR10_data_prep -> fn

|   ├── S10_Tinyimagenet_data_prep -> fn

|   ├── create_tensorboard_writer -> fn

|   ├── GradCAM -> Class

|   	├── __init__() -> fn

|   	├── _encode_one_hot() -> fn

|   	├── forward() -> fn

|   	├── backward() -> fn

|   	├── remove_hook() -> fn

|   	├── _find() -> fn

|   	├── generate() -> fn

|   ├── GRADCAM -> fn

|   ├── LRRangeFinder -> Class

|   	├── __init__() -> fn

|   	├── findLR() -> fn

|   ├── plot_onecyclelr_curve -> fn

|   ├── TinyImagenetHelper -> Class

|   	├── __init__() -> fn

|   	├── get_id_dictionary() -> fn

|   	├── get_train_test_labels_data() -> fn

|   ├── TinyImagenetDataset -> Class

|   	├── __init__() -> fn

|   	├── __len__() -> fn

|   	├── __getitem__() -> fn

|   	├── is_grayscale_image() -> fn

|   ├── Tinyimagenet_Dataloader -> Class

|   	├── __init__() -> fn

|   	├── gettraindataloader() -> fn

|   	├── gettestdataloader() -> fn 

├── README.md 
