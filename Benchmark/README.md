Summary of Key Takeaways

* Start with clear objectives (Understand what does `Similar` mean for our case).
* Select or create a robust dataset with right balance of diversity and specificity.
* Annotate throughly to establish ground truth .
* Choose metric that reflect both ranking quality and useer needs (mAP, P@K, nDCG).
* Analyze results in detail to pinpoint weaknesses and guide further system improvements.


**Similarity in our case** 
In out case similarity is shapes and colors of item. Becouse we already know searchabel categroy (sink, tub, cabinet ect.). In short words Retrieval based on certain attributes (color, texture, style).About the scope I want to make it not larger than 20**4 iterm for all dataset which will include all categoryes.The user is ecommerce customer looking for similar items for his porduct.

**Dataset Selection**
** So best way to start is public Banchmark Datasets. ** 
* Flickr8k/Flickr30k (general images, including people, animals, and objects).
* COCO (Common Objects in Context) for a broader set of labeled images.
* Oxford and Paris Datasets for landmark retrieval.
* Fashion-MNIST or DeepFashion for fashion/clothing-based similarit