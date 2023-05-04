# Chapter 15: Tweedledee and Tweedledum: A Look at Ensemble Learning

_Once upon a time in the magical world of DataLand, our heroine Alice embarked upon an exciting new adventure. Little did she know that upon her arrival at a certain point on the yellow brick road, she would uncover the secrets of an extraordinary mathematical technique, one that is all aflutter with wondrous possibilities - Ensemble Learning!_

In a surprising turn of events, Alice stumbles upon Tweedledee and Tweedledum, an extraordinary pair of data scientists who are not just curious but quick-witted companions. They play a vital role in showing Alice how to harness the power of multiple learning algorithms together to achieve better predictions and overall performance. Brace yourselves, dear Python wizards, for a delightful journey into the whimsical world of Ensemble Learning!

Now, what do we have here? :thought_balloon:

Well, Ensemble Learning is a delightful technique that combines the predictions from multiple learning algorithms to decide the output more accurately. In the magical storybooks of world-renowned researchers, we read that Ensemble Learning can help us achieve better results by leveraging the strengths of various algorithms ([Dietterich, 2000](https://ieeexplore.ieee.org/document/895284)). Let's dive deeper into the wonders of Ensemble Learning, guided by our quirky companions, Tweedledee and Tweedledum.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

:star2: Ensemble Learning is much like a wise council of elders. They say there's wisdom in numbers, and so it is with Ensemble Learning methods! Just as our Pythonista Alice learns from her curious companions, the power of Ensemble Learning lies in the combination of different learners with their unique abilities.

Clap your hands and join us in the enchanting village of DataLand. It's time to explore a few famous Ensemble Learning Methods!

## 1. Bagging

_Tighten your boots and backpacks, dear adventurer, as we embark upon a Bagging quest that lives in the realm of Parallel Ensemble Methods._ :evergreen_tree:

Bagging, which stands for Bootstrap Aggregating, involves the construction of many different models in parallel by training each classifier independently. This isolation of classifiers reduces the risk of overfitting and provides a good mixture of smart decisions ([Breiman, 1996](https://link.springer.com/article/10.1023%2FA%3A1018054314350)).

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Create a BaggingClassifier object
bagging = BaggingClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=42)

# Train and evaluate the model
bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)
print("Bagging Accuracy:", accuracy_score(y_test, y_pred_bagging))
```

## 2. Boosting

_Onward we go, higher and higher, for it's time to Boost our model's ability through the power of Sequential Ensemble Methods._ :mountain:

Boosting transforms a group of weak learners, such as decision trees, into strong learners by iteratively training them, focusing on the examples that are harder to classify ([Friedman et al., 2000](https://projecteuclid.org/euclid.aos/1013699998)). Each weak learner has the opportunity to correct the mistakes its predecessors made, increasing the predictive power of the ensemble.

```python
from sklearn.ensemble import AdaBoostClassifier

# Create an AdaBoostClassifier object
adaboost = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=42)

# Train and evaluate the model
adaboost.fit(X_train, y_train)
y_pred_adaboost = adaboost.predict(X_test)
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_adaboost))
```

## 3. Stacking

_And now, dear wanderer, let the magical art of Stacking whisk you away! For here lies the secret of Metaclassifiers guiding the congregation of base learners._ :sparkles:

Stacking is an ensemble technique that combines the predictions from multiple models by training a metaclassifier on the predictions themselves ([Wolpert, 1992](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.56.1533)). The metaclassifier reaches better results by learning how to best combine the differing expertise of various base models.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC

# Create a StackingClassifier object
stack = StackingClassifier([('Decision Tree', DecisionTreeClassifier()),
                            ('SVC', SVC())],
                           RandomForestClassifier(n_jobs=-1, random_state=42))

# Train and evaluate the model
stack.fit(X_train, y_train)
y_pred_stack = stack.predict(X_test)
print("Stacking Accuracy:", accuracy_score(y_test, y_pred_stack))
```

Our adventure in DataLand continues to be full of surprises and wonder. Keep sight of the horizon and join us in the exploration of Ensemble Learning guided by the playful duo, Tweedledee and Tweedledum. Let us continue, for there are many more secrets to uncover in Alice's Adventure in Deep Learning Mathematics for Python Wizards! :books:
# Chapter 15: Tweedledee and Tweedledum: The Ensemble Learning Adventure

_In the wondrous world of DataLand, Alice traverses the vibrant landscape, ever so enchanted by the grandeur of it all. As she makes her way past the towering mushroom forests and fragrant flower-filled fields, she is greeted by a curious sight -- Tweedledee and Tweedledum!_

_Wait, what's this? The riddle-solving pair, Tweedledee and Tweedledum, huddled together around a rather intriguing puzzle. Of course, it had something to do with the world of DataLand! Eager to learn, Alice can't help but join them._

## The Bagging Party :tada:

The Tweedles led Alice to a grand gathering in the misty meadows, and behold! A Bagging party is in progress. Friendly baggers are dancing in circles, creating multiple subsets from a single dataset.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging = BaggingClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=42)
```

_Exchanging courteous nods, the baggers carry their own data subsets as they weave through the rotating vortex. They create numerous Decision Trees, each learning from their individual bags of data. As they swing and sway in perfect harmony, their combined wisdom outshines individual mistakes, and out emerges a strong classifier._

_Alice claps her hands in delight as she watches the baggers twirl around, and with Tweedledee and Tweedledum's guidance, she records the accuracy of their marvelous dance._

```python
bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)
print("Bagging Accuracy:", accuracy_score(y_test, y_pred_bagging))
```

## The Boosting Parade :balloon:

Next, Alice follows the Tweedles to witness the Boosting parade, where a group of weak learners are marching forward. Everyone seems so eager to learn! Step by step, each weak learner tries to overcome the errors made by the previous one.

```python
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=42)
```

Suddenly, the parade lights up with brilliant sparks, as Alice sees how each learner becomes stronger by following the path of the one before it. The ingenious Tweedledee whispers that this fascinating phenomenon is called "Adaptive Boosting," which strengthens weak learners sequentially!

Feeling more confident with each boost, Alice quickly checks the accuracy of this glowing parade.

```python
adaboost.fit(X_train, y_train)
y_pred_adaboost = adaboost.predict(X_test)
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_adaboost))
```

## The Stacking Palace :crown:

The adventure is far from over! The Tweedles, full of energy, take Alice to the Stacking Palace, where base learners join forces in regal chambers. There, a "metaclassifier" was orchestrating their decisions from a throne.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC

stack = StackingClassifier([('Decision Tree', DecisionTreeClassifier()),
                            ('SVC', SVC())],
                           RandomForestClassifier(n_jobs=-1, random_state=42))
```

The delegation of base learners relays their decisions to the wise metaclassifier, which then weaves together a combined prediction worthy of its royal mantle. Alice watches intently, observing how the metaclassifier harmonizes the differing opinions to create a stronger, more accurate verdict.

Triumphant and truly astonished, Alice checks the accuracy of this formidable Stacking assembly.

```python
stack.fit(X_train, y_train)
y_pred_stack = stack.predict(X_test)
print("Stacking Accuracy:", accuracy_score(y_test, y_pred_stack))
```

_Already looking forward to her next adventure, Alice cherishes the wisdom she gained from her delightful encounters with Tweedledee and Tweedledum. This land of DataLand's Ensemble Learning methods will remain forever etched in her memories, and she can't help but feel ever-so-grateful for the exciting journey she just undertook._

_With a bright future ahead for Alice in DataLand, stay tuned for the upcoming challenges that await her among the depths of Deep Learning Mathematics, where Python Wizards like Alice lead the way._ :sparkling_heart:
## Code Explanation: Ensemble Learning Adventure

In the whimsical adventure that Alice embarked upon, she discovered the beautiful world of Ensemble Learning. Here, she witnessed the beauty of Bagging, the intrigue of Boosting, and the grandeur of Stacking. Let's unwrap each of these ensemble methods and explore the code snippets used in the story.

### Bagging

#### Code:
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging = BaggingClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=42)
```

In this segment, Bagging is implemented using the `BaggingClassifier` from the `sklearn.ensemble` module. The base estimator used here is a `DecisionTreeClassifier`. The `n_estimators=10` parameter tells the classifier to create 10 Decision Trees, each trained on a random subset of the original dataset. The `random_state` parameter is set for reproducibility of the results.

```python
bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)
print("Bagging Accuracy:", accuracy_score(y_test, y_pred_bagging))
```

Here, Alice fits the Bagging model to her training data (`X_train`, `y_train`). Once fitting is complete, she predicts the test data (`X_test`). Finally, she calculates and prints the accuracy of the Bagging ensemble using the `accuracy_score` function.

### Boosting (Adaptive Boosting)

#### Code:
```python
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=42)
```

In this segment, Adaptive Boosting (AdaBoost) is implemented using the `AdaBoostClassifier` from the `sklearn.ensemble` module. Similar to Bagging, the base estimator is a `DecisionTreeClassifier` and `n_estimators` is set to 10. The `random_state` parameter is set for reproducibility.

```python
adaboost.fit(X_train, y_train)
y_pred_adaboost = adaboost.predict(X_test)
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_adaboost))
```

Here, Alice fits the AdaBoost model to her training data and makes predictions on the test data. She then calculates and prints the accuracy of the AdaBoost ensemble.

### Stacking

#### Code:
```python
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC

stack = StackingClassifier([('Decision Tree', DecisionTreeClassifier()),
                            ('SVC', SVC())],
                           RandomForestClassifier(n_jobs=-1, random_state=42))
```

In this segment, Stacking is achieved using the `StackingClassifier` from the `sklearn.ensemble` module. Alice stacks a `DecisionTreeClassifier` along with a Support Vector Classifier (`SVC`). She uses the `RandomForestClassifier` with parallel processing (`n_jobs=-1`) as a metaclassifier, coordinating the base models to create a harmonized prediction. The `random_state` is set for reproducibility.

```python
stack.fit(X_train, y_train)
y_pred_stack = stack.predict(X_test)
print("Stacking Accuracy:", accuracy_score(y_test, y_pred_stack))
```

Finally, Alice fits the Stacking model to the training data and makes predictions on the test data. She calculates and prints the accuracy of the Stacking ensemble to complete her thrilling adventure with the Tweedle twins.