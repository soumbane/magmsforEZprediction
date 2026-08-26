#!/bin/bash

# Shard right0 of the right hemisphere (78 nodes) on cuda:0
# Trains on the original SMOTE-augmented training data (58 patients) with all 5 sequences and
# cross-sequence distillation, selecting the best checkpoint on the 17 subjects of the additional
# cohort that have all five sequences. Runs sequentially in the foreground; use
# ./run_all_add_training.sh to launch every shard detached so the run survives closing SSH.

node_nums=(504 506 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 524 525 526 529 530 534 535 536 537 538 539 540 541 542 543 546 547 548 549 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 581 582 584 585 586 587 588 589 590 591 592 593 594 595 596 598)

# Loop through each node_num
for node_num in "${node_nums[@]}"; do
    # Define experiment file path
    exp_file="exp_node${node_num}/AddCohort/magms"

    # Run the training script with specified arguments
    python train_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_trained_last_righthemis.model -b 4 -lr 1e-2 --num_mod 5 --node_num ${node_num} --train_mod ALL -e 30 -exp ${exp_file} --replace_experiment --show_verbose --device cuda:0

    # Record the experiment file
    echo "Experiment for node_num ${node_num} saved at: ${exp_file}"

done
