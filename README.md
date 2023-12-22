# Calculator of handwritten numbers
In this research i've developed and fitted a simple CNN model which detects handwritten digits and some mathematical symbols (decimal, sum, substract, multiply and divide).
For that I've used [this dataset](https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols/data).
Then I've put fitted model and expression recognition method in telegram bot.

![Illustration 1](https://github.com/sancteBaphometh/ai_calculator/blob/main/readme%20images/5.png)

For that task i've developed a really simple and lightweight architecture based on CNN. At the last layer there is sigmoid actiation function for multiclass classification.
I've fitted it for 25 epochs with scheduler and augmentations and got 0.98 accuracy on validation set.

The model's arcitecture is presented at the image:

![Illustration 2](https://github.com/sancteBaphometh/ai_calculator/blob/main/readme%20images/7.png)

Fitting results below.

Accuracy:

![Illustration 3](https://github.com/sancteBaphometh/ai_calculator/blob/main/readme%20images/6.png)

Precision/recall, f1-score for every class:

![Illustration 4](https://github.com/sancteBaphometh/ai_calculator/blob/main/readme%20images/8.png)

---

### Results and test

Below you can check out results of using model on examples.

![Illustration 5](https://github.com/sancteBaphometh/ai_calculator/blob/main/readme%20images/1.png)

![Illustration 6](https://github.com/sancteBaphometh/ai_calculator/blob/main/readme%20images/2.png)

![Illustration 7](https://github.com/sancteBaphometh/ai_calculator/blob/main/readme%20images/3.png)

![Illustration 8](https://github.com/sancteBaphometh/ai_calculator/blob/main/readme%20images/4.png)

As you can see model's doing not so well as I've wanted it to. It's abolutely can't detect decimals and it does a ton of mistakes while detecting divide symbol.

I think this is a expression detection method's problem and it's needed to remake this method in orded to bot worked well because as you could see in statistics the model is good with single symbols.
