# 任务一 典型错误案例（用于报告配图）

> 从 `outputs/task1/error_cases/error_cases.csv`（最显著的 200 例）中按四类错误各挑选 3 例。原图已下载到 `reports/figures/error_cases/`。

> 说明：自动标志为启发式初筛，以下为人工挑选的可解释样例；同一图可能同时命中多类错误。


## 食材误识别（`ingredient_misidentification`）

### 案例 1　（class: `unlabeled`）

- 原图：`reports/figures/error_cases/images__test__unlabeled__000170.jpg`（image_id: `images/test/unlabeled/000170.jpg`）
- 参考描述：Baked apple strudel with almond slices on the surface on a white plate.
- ResNet18+GRU：a piece of pumpkin cake with a super brown crust and a piece of brown bread on a white plate
- ViT+Transformer：a large piece of apple strudel with thin yellow apple pieces and white sugar
  - ResNet18+GRU 要点：缺: almond, apple, strudel；多: bread, cake, pumpkin, super；烹饪: pred=|ref=baked
  - ViT+Transformer 要点：缺: almond；多: sugar；烹饪: pred=|ref=baked

### 案例 2　（class: `unlabeled`）

- 原图：`reports/figures/error_cases/images__test__unlabeled__000307.jpg`（image_id: `images/test/unlabeled/000307.jpg`）
- 参考描述：Some bone-in grilled Ayam goreng seasoned with various fresh spices.
- ResNet18+GRU：a plate of ayam goreng with a crispy skin served with a small piece of beer
- ViT+Transformer：three pieces of ayam goreng on a white plate
  - ResNet18+GRU 要点：缺: spices；多: beer；烹饪: pred=|ref=grilled
  - ViT+Transformer 要点：缺: spices；烹饪: pred=|ref=grilled

### 案例 3　（class: `unlabeled`）

- 原图：`reports/figures/error_cases/images__test__unlabeled__000008.jpg`（image_id: `images/test/unlabeled/000008.jpg`）
- 参考描述：A large slice of oily grilled AEbleflaesk in combination with mashed apples
- ResNet18+GRU：a large plate of aebleflaesk with a lot of pork belly and pork in a brown sauce
- ViT+Transformer：a thick piece of aebleflaesk with crispy apple pieces and apple pieces and apple pieces
  - ResNet18+GRU 要点：缺: apples, mashed；多: belly, pork；烹饪: pred=|ref=grilled
  - ViT+Transformer 要点：缺: apples, mashed；多: apple；烹饪: pred=|ref=grilled


## 烹饪方式错误（`cooking_method_error`）

### 案例 1　（class: `unlabeled`）

- 原图：`reports/figures/error_cases/images__test__unlabeled__000053.jpg`（image_id: `images/test/unlabeled/000053.jpg`）
- 参考描述：Big thin round almond biscuits baked brown and stacked on a square white plate.
- ResNet18+GRU：some golden corn dogs stacked together on a white plate
- ViT+Transformer：a stack of golden brown fried corn tortillas stacked together
  - ResNet18+GRU 要点：缺: almond, biscuits；多: corn, dogs, together；烹饪: pred=|ref=baked
  - ViT+Transformer 要点：缺: almond, biscuits；多: corn, stack, together, tortillas；烹饪: pred=fried|ref=baked

### 案例 2　（class: `unlabeled`）

- 原图：`reports/figures/error_cases/images__test__unlabeled__000443.jpg`（image_id: `images/test/unlabeled/000443.jpg`）
- 参考描述：steamed banh bo with spongy skin tined into green and purple.
- ResNet18+GRU：many white banh bos are placed on a white plate
- ViT+Transformer：a pile of freshly baked banh bo cakes in the middle and grated coconut
  - ResNet18+GRU 要点：缺: bo, purple, spongy；多: placed；烹饪: pred=|ref=steamed
  - ViT+Transformer 要点：缺: purple, spongy；多: cakes, coconut, freshly, grated；烹饪: pred=baked|ref=steamed

### 案例 3　（class: `unlabeled`）

- 原图：`reports/figures/error_cases/images__test__unlabeled__000032.jpg`（image_id: `images/test/unlabeled/000032.jpg`）
- 参考描述：Pan-fried agedashi tofu with a light brown surface topped with some chopped green onions in soy sauce.
- ResNet18+GRU：a plate of juicy agedashi tofu with chopped spring onion and sprinkled with chopped spring onions
- ViT+Transformer：a plate of svickova consists of some braised pork belly slices and topped with some fried potato wedges and finely chopped green onions slices
  - ResNet18+GRU 要点：缺: soy；多: juicy, onion, spring, sprinkled；烹饪: pred=|ref=pan-fried
  - ViT+Transformer 要点：缺: agedashi, soy, tofu；多: belly, consists, finely, pork；烹饪: pred=braised,fried|ref=pan-fried


## 食材漏述（`missing_ingredient`）

### 案例 1　（class: `unlabeled`）

- 原图：`reports/figures/error_cases/images__test__unlabeled__000579.jpg`（image_id: `images/test/unlabeled/000579.jpg`）
- 参考描述：rice adding bistek full of thin fried beef and onion rings on a square plate.
- ResNet18+GRU：a plate of bistek with some beef slices and some green onion rings on the top of the plate
- ViT+Transformer：a plate of bistek with onions onions and shredded onions
  - ResNet18+GRU 要点：缺: rice；烹饪: pred=|ref=fried
  - ViT+Transformer 要点：缺: beef, onion, rice, rings；多: onions, shredded；烹饪: pred=|ref=fried

### 案例 2　（class: `unlabeled`）

- 原图：`reports/figures/error_cases/images__test__unlabeled__000131.jpg`（image_id: `images/test/unlabeled/000131.jpg`）
- 参考描述：Apple crisps covered with flour baked into brown and placed on a white round plate.
- ResNet18+GRU：a white plate of golden apple sauce with a thick and white sauce on the side
- ViT+Transformer：a plate of jambalaya made of rice shrimps and potatoes slices and a spoon placed on a white round plate
  - ResNet18+GRU 要点：缺: crisps, flour, placed；烹饪: pred=|ref=baked
  - ViT+Transformer 要点：缺: apple, crisps, flour；多: jambalaya, potatoes, rice, shrimps；烹饪: pred=|ref=baked

### 案例 3　（class: `unlabeled`）

- 原图：`reports/figures/error_cases/images__test__unlabeled__000430.jpg`（image_id: `images/test/unlabeled/000430.jpg`）
- 参考描述：lightly baked bangers with sliced onions on the top of mashed potatoes.
- ResNet18+GRU：a plate of pork knuckle with a brown sauce and a piece of pork on the side
- ViT+Transformer：a plate of bangers and mash with some onion sauce and some mashed potatoes on the top and some mashed potatoes on the bottom
  - ResNet18+GRU 要点：缺: bangers, mashed, onions, potatoes；多: knuckle, pork；烹饪: pred=|ref=baked
  - ViT+Transformer 要点：缺: onions；多: mash, onion；烹饪: pred=|ref=baked


## 无依据添加（`unsupported_addition`）

### 案例 1　（class: `unlabeled`）

- 原图：`reports/figures/error_cases/images__test__unlabeled__000368.jpg`（image_id: `images/test/unlabeled/000368.jpg`）
- 参考描述：A white oven of baked beans with reddish brown color and moist texture.
- ResNet18+GRU：a bowl of dark brown beans with a brown texture
- ViT+Transformer：a plate of dark red beans with a lot of meat sauce in it in a white round bowl
  - ResNet18+GRU 要点：烹饪: pred=|ref=baked
  - ViT+Transformer 要点：多: meat；烹饪: pred=|ref=baked

### 案例 2　（class: `unlabeled`）

- 原图：`reports/figures/error_cases/images__test__unlabeled__000000.jpg`（image_id: `images/test/unlabeled/000000.jpg`）
- 参考描述：Baked thin AEbleflaesk slices with brown dried surface alternate with yellow apple slices.
- ResNet18+GRU：a plate of aebleflaesk with a lot of pork belly and yellow apple slices
- ViT+Transformer：a plate of aebleflaesk with red thin slices of brown pork belly and a piece of brown bread
  - ResNet18+GRU 要点：多: belly, pork；烹饪: pred=|ref=baked,dried
  - ViT+Transformer 要点：缺: apple；多: belly, bread, pork；烹饪: pred=|ref=baked,dried

### 案例 3　（class: `unlabeled`）

- 原图：`reports/figures/error_cases/images__test__unlabeled__000490.jpg`（image_id: `images/test/unlabeled/000490.jpg`）
- 参考描述：a whole roasted beef with cracks on the surface coated with a pasty served on a plate.
- ResNet18+GRU：a piece of beef pie with a thick and crispy skin on the surface and a piece of bread
- ViT+Transformer：a piece of brown applesauce cake with black raisins and cinnamon inside and a piece of chocolate on the top in a plate
  - ResNet18+GRU 要点：缺: coated；多: bread, pie；烹饪: pred=|ref=roasted
  - ViT+Transformer 要点：缺: beef, coated；多: applesauce, cake, chocolate, cinnamon；烹饪: pred=|ref=roasted

