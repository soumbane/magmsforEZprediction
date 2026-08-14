#!/bin/bash

# Shard left1 of the left hemisphere (89 nodes) on cuda:2
# Evaluates the AddCohort checkpoints on both groups of the additional cohort:
#   - the 17 subjects with all five sequences, on all 31 modality combinations
#   - the 13 subjects without DWIC, on the 15 DWIC-free combinations
# Each of the 3 training trials is written as its own column. For a detached run use
# eval_sh_scripts/nh_eval_add_left1.sh

node_nums=(205 211 213 214 216 217 220 221 222 224 225 226 227 228 229 230 231 232 233 234 235 238 239 240 241 245 246 247 248 251 252 253 256 257 260 261 275 290 291 292 294 295 296 297 298 299 301 302 303 304 305 306 316 320 321 322 326 331 332 334 335 336 337 338 339 340 343 346 349 352 353 354 355 356 357 359 360 361 362 363 364 365 366 367 368 369 370 371 372)

# Loop through each node_num
for node_num in "${node_nums[@]}"
do
    # Run the evaluation script with specified arguments
    python eval_left.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/ -b 4 --node_num ${node_num} --replace_experiment --show_verbose --device cuda:2

    # Record the experiment file
    echo "Done Evaluating ALL modalities for node_num ${node_num}"
done
