"""
File contains codes for the tests of the methods on different metrics.
To test a method, just copy the part relative to your method and paste in Google Colab where needed.

The lines were commented in order to not have errors since these were written for Colab not the local PC.
Uncomment the one you want to use.
"""

# THE PART FOR EVERYONE STARTS HERE

methods = ["netvlad", "cosplace", "mixvpr", "megaloc"]
distance_metrics = ["L2", "dotproduct"]

# The prof said that for the svox dataset, we have to use "gallery" as database and use "queries_night" and "queries_sun" as queries for it
# This dictionary associates to each database, its particular queries to be used with it
distance_metrics = f"L2 dotproduct"
databases = f"/content/data/sf_xs/test/database /content/data/tokyo_xs/test/database /content/data/svox/images/test/gallery /content/data/svox/images/test/gallery"
queries = f"/content/data/sf_xs/test/queries /content/data/tokyo_xs/test/queries /content/data/svox/images/test/queries_night /content/data/svox/images/test/queries_sun"

# THE PART FOR  EVERYONE ENDS HERE
#   UNCOMMENT AND PASTE IN COLAB IF YOU WANT TO USE NETVLAD
"""
!python /content/Visual-Place-Recognition-Project/VPR-methods-evaluation/main.py \
--num_workers 8 \
--batch_size 32 \
--log_dir log_dir \
--method=netvlad --backbone=VGG16 --descriptors_dimension=4096 \
--image_size 480 640 \
--database_folder $databases \
--queries_folder $queries \
--num_preds_to_save 20 \
--recall_values 20 \
--distance_metric $distance_metrics
"""
#   UNCOMMENT AND PASTE IN COLAB IF YOU WANT TO USE COSPLACE
"""
!python /content/Visual-Place-Recognition-Project/VPR-methods-evaluation/main.py \
--num_workers 8 \
--batch_size 32 \
--log_dir log_dir \
--method=cosplace --backbone=ResNet18 --descriptors_dimension=512 \
--image_size 512 512 \
--database_folder $database \
--queries_folder $queries \
--num_preds_to_save 20 \
--recall_values 20 \
--distance_metric=$distance_metric
"""
#   UNCOMMENT AND PASTE IN COLAB IF YOU WANT TO USE MIXVPR
"""
!python /content/Visual-Place-Recognition-Project/VPR-methods-evaluation/main.py \
--num_workers 8 \
--batch_size 32 \
--log_dir log_dir \
--method=mixvpr --backbone=ResNet50 --descriptors_dimension=512 \
--image_size 320 320 \
--database_folder $database \
--queries_folder $queries \
--num_preds_to_save 20 \
--recall_values 20 \
--distance_metric=$distance_metric
"""

#   UNCOMMENT AND PASTE IN COLAB IF YOU WANT TO USE MEGALOC
"""
!python /content/Visual-Place-Recognition-Project/VPR-methods-evaluation/main.py \
--num_workers 8 \
--batch_size 32 \
--log_dir log_dir \
--method=megaloc --backbone=Dinov2 --descriptors_dimension=8448 \
--image_size 322 322 \
--database_folder $database \
--queries_folder $queries \
--num_preds_to_save 20 \
--recall_values 20 \
--distance_metric=$distance_metric
"""