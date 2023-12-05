#!/bin/bash

# Define the list of node_num values
node_nums=(291 292 294 295 296 297 298 299 301 302 303 304 305 306 316 320 321 322 326 331 332 334 335 336 337 338 339 340 343 346 349 352 353 354 355 356 357 359 360 361 362 363 364 365 366 367 368 369 370 371 372 374 375 376 377 378 381 382 383 384 386 387 388 389 390 391 394 395 396 397 398 399 400 401 402 403 404 405 406 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 426 427 428 429 430 431 432 433 435 436 437 438 439 440 441 442 443 444 445 446 447 448 450 451 452 453 454 455 456 458 459 460 461 462 463 464 465 466 467 469 470 471 472 473 474 475 476 477 478 479)  # part 2 - left hemis

# Loop through each node_num
for node_num in "${node_nums[@]}"
do    
    # Run the evaluation script with specified arguments
    python eval_left.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/ -b 4 --node_num ${node_num} --replace_experiment --show_verbose --device cuda:1

    # Record the experiment file
    echo "Done Evaluating ALL modalities for node_num ${node_num}"
done