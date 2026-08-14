#!/bin/bash

# Shard right0 of the right hemisphere (89 nodes) on cuda:1
# Evaluates the AddCohort checkpoints on both groups of the additional cohort:
#   - the 17 subjects with all five sequences, on all 31 modality combinations
#   - the 13 subjects without DWIC, on the 15 DWIC-free combinations
# Each of the 3 training trials is written as its own column. For a detached run use
# eval_sh_scripts/nh_eval_add_right0.sh

node_nums=(504 506 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 524 525 526 529 530 534 535 536 537 538 539 540 541 542 543 546 547 548 549 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 581 582 584 585 586 587 588 589 590 591 592 593 594 595 596 598 599 600 601 602 603 604 605 606 607 608 609)

# Loop through each node_num
for node_num in "${node_nums[@]}"
do
    # Run the evaluation script with specified arguments
    python eval_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/ -b 4 --node_num ${node_num} --replace_experiment --show_verbose --device cuda:1

    # Record the experiment file
    echo "Done Evaluating ALL modalities for node_num ${node_num}"
done
