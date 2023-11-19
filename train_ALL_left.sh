#!/bin/bash

# Define the list of node_num values
# node_nums=(385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479) # part 1
node_nums=(386 387 388 389 390 391 394 395 396 397 398 399 400 401 402 403 404 405 406 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 426 427 428 429 430 431 432 433 435 436 437 438 439 440 441 442 443 444 445 446 447 448 450 451 452 453 454 455 456 458 459 460 461 462 463 464 465 466 467 469 470 471 472 473 474 475 476 477 478 479)  # part 2

# Loop through each node_num
for node_num in "${node_nums[@]}"
do
    # Define experiment file path
    # exp_file="exp_node${node_num}/NO_Distillation/magms"
    exp_file="exp_node${node_num}/Part_2/magms"

    # Run the training script with specified arguments
    python train_left.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Temporal_Lobe/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_trained_last_lefttemp.model -b 4 -lr 1e-2 --num_mod 5 --node_num ${node_num} --train_mod ALL -e 30 -exp ${exp_file} --replace_experiment --show_verbose --device cuda:0

    # Record the experiment file
    echo "Experiment for node_num ${node_num} saved at: ${exp_file}"
    
done

