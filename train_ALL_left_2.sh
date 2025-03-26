#!/bin/bash

node_nums=(320 321 322 326 331 332 334 335 336 337 338 339 340 343 346 349 352 353 354 355 356 357 359 360 361 362 363 364 365 366 367 368 369 370 371 372 374 375 376 377 378 381 382 383 384 386 387 388 389 390 391 394 395 396 397 398 399 400 401 402 403 404 405 406 408 409 410 411 412)  # part 2 - left hemis

# Loop through each node_num
for node_num in "${node_nums[@]}"; do    
    # Define experiment file path
    exp_file="exp_node${node_num}/NO_Distillation/magms"
    # exp_file="exp_node${node_num}/Part_2/magms"

    # Run the training script with specified arguments
    python train_left.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_trained_last_lefthemis.model -b 4 -lr 1e-2 --num_mod 3 --node_num ${node_num} --train_mod T1-T2-FLAIR -e 30 -exp ${exp_file} --replace_experiment --show_verbose --device cuda:1

    # Record the experiment file
    echo "Experiment for node_num ${node_num} saved at: ${exp_file}"
    
done

