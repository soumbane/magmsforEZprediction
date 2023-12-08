#!/bin/bash

# Define the list of node_num values
node_nums=(165 166 169 175 176 177 192 193 194 195 197 198 199 200 202 204 205 211 213 214 216 217 220 221 222 224 225 226 227 228 229 230 231 232 233 234 235 238 239 240 241 245 246 247 248 251 252 253 256 257 260 261 275 290 291 292 294 295 296 297 298 299 301 302 303 304 305 306 316 320 321 322)  # part 2 - left hemis

# Loop through each node_num
for node_num in "${node_nums[@]}"
do    
    # Run the evaluation script with specified arguments
    python eval_left.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/ -b 4 --node_num ${node_num} --replace_experiment --show_verbose --device cuda:0

    # Record the experiment file
    echo "Done Evaluating ALL modalities for node_num ${node_num}"
done