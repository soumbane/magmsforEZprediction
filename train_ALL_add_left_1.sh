#!/bin/bash

# Shard left1 of the left hemisphere (73 nodes) on cuda:0
# Trains on the original SMOTE-augmented training data (58 patients) with all 5 sequences and
# cross-sequence distillation, selecting the best checkpoint on the 17 subjects of the additional
# cohort that have all five sequences. Runs sequentially in the foreground; for a detached run use
# train_sh_scripts/nh_train_add_left1.sh

node_nums=(159 160 163 164 165 166 169 175 176 177 192 193 194 195 197 198 199 200 202 204 205 211 213 214 216 217 220 221 222 224 225 226 227 228 229 230 231 232 233 234 235 238 239 240 241 245 246 247 248 251 252 253 256 257 260 261 275 290 291 292 294 295 296 297 298 299 301 302 303 304 305 306 316)

# Loop through each node_num
for node_num in "${node_nums[@]}"; do
    # Define experiment file path
    exp_file="exp_node${node_num}/AddCohort/magms"

    # Run the training script with specified arguments
    python train_left.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_trained_last_lefthemis.model -b 4 -lr 1e-2 --num_mod 5 --node_num ${node_num} --train_mod ALL -e 30 -exp ${exp_file} --replace_experiment --show_verbose --device cuda:0

    # Record the experiment file
    echo "Experiment for node_num ${node_num} saved at: ${exp_file}"

done
