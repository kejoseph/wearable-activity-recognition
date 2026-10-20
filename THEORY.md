# Theory Guide: Activity Recognition from Wearable Sensors

A study guide for building this project. It explains **why** each step matters
and what to look for in your results. It does not give you the model code: that
part is yours to write.

Work through the sections in order. Each ends with **Check yourself** questions;
if you can answer them out loud, you can answer them in an interview.

---

## 1. Frame the problem

- **Task:** multiclass classification. Input: one 2.56 s window of sensor data.
  Output: one of 6 activities (walking, upstairs, downstairs, sitting, standing, laying).
- **Product framing:** this is the model behind the "you've been sitting for an
  hour" nudge or the automatic workout detection on a watch. Keep asking: what
  happens to a *user* when the model is wrong?

**Check yourself**
- Which mistake is worse for a fitness product: calling a walk "sitting", or calling sitting "walking"? Why?

---

## 2. Know the data's structure before modeling

- **Windows:** signals sampled at 50 Hz, cut into 128-sample windows with **50% overlap**.
  Neighboring windows share half their data.
- **Subjects:** 30 people. Every window belongs to exactly one person.
- **Two representations of the same windows:**
  - `features`: 561 hand-engineered features (means, standard deviations, FFT energy, etc.)
  - `raw`: 9 channels x 128 timesteps (body acceleration, gyroscope, total acceleration on x/y/z)
- **Gravity:** `total_acc` includes gravity; `body_acc` has gravity removed. Gravity's
  direction tells you the phone's orientation, which matters a lot for static postures.

**Check yourself**
- Why does 50% overlap matter for how you split data? (Answer in section 3.)
- Which channels would you expect to separate *laying* from *standing*? Which separate *walking* from *walking upstairs*?

---

## 3. Leakage: the most important lesson in this project

**The trap:** if you shuffle all windows and split randomly, the same person
appears in both train and test, and overlapping windows put *nearly identical
data* on both sides. The model partly memorizes people instead of learning activities.
The score looks great and is wrong.

**The fix:** split **by subject**. Every window from a person goes entirely into
train or entirely into test. In scikit-learn terms, this is grouped cross-validation
with `subject` as the group.

**The experiment to run (and put in your README):**
1. Score a model with random K-fold.
2. Score the same model with subject-grouped K-fold.
3. Report the gap. That gap is the size of the leak.

**What it means for a product:** a real user is always a *new person* the model
never saw. Only the grouped score estimates what users will experience.

**Leakage hides in other places too:**
- Fitting a scaler or normalization stats on all data before splitting (fit on train folds only)
- Choosing hyperparameters using the test set
- Early stopping on a validation set that shares subjects with the training set

**Check yourself**
- Explain to a non-technical manager why the random-split accuracy is misleading.
- Your grouped score is 4 points lower than random. Is your model worse? (No: your *estimate* is more honest.)

---

## 4. Baselines first

- **Majority-class baseline:** always predict the most common activity. This is the
  floor. A model is only interesting by how much it beats this.
- **Simple model before complex model:** if logistic regression gets 95%, a CNN at 96%
  needs a strong reason (cost, latency, maintenance) to be worth it.

**Check yourself**
- Why might a stakeholder be impressed by "90% accuracy" when it shouldn't be? (Think class balance.)

---

## 5. Choose the right metrics

- **Accuracy:** fine when classes are roughly balanced (they are here), but it hides
  which classes fail.
- **Macro-F1:** averages F1 across classes equally, so a weak minority class can't hide.
  Report it alongside accuracy.
- **Per-class precision and recall:**
  - *Precision* for "sitting": of the times we said sitting, how often was it sitting?
  - *Recall* for "sitting": of all real sitting, how much did we catch?
- **Confusion matrix:** the single most useful plot. It shows *which* classes get confused.
- **Spread, not just the average:** report mean +/- standard deviation across folds,
  and look at the score **per subject**. The worst-served user matters for a product.

**Check yourself**
- When would you optimize for recall over precision for one activity?
- Your mean accuracy is 93% but one subject scores 70%. What might be different about that person?

---

## 6. The model ladder

### 6a. Logistic regression (on `features`)
- A **linear** model: a weighted sum of features per class, turned into probabilities with softmax.
- **Scale features** first (it's sensitive to feature magnitude; trees are not).
- **Regularization (`C`):** smaller `C` = stronger penalty on large weights = simpler model.
  With 561 correlated features, regularization prevents overfitting.
- Why it can do well here: the 561 features were engineered by experts, so much of the
  hard work is already done.

### 6b. Gradient boosting (LightGBM, on `features`)
- Builds many small decision trees in sequence; each tree corrects the previous ones' errors.
- Captures **non-linear** interactions without feature scaling.
- Key knobs: number of trees, learning rate (smaller = needs more trees, generalizes better),
  tree size (`num_leaves`), and subsampling rows/columns (adds randomness, reduces overfitting).
- Gives **feature importances**: useful for explaining *what* the model relies on.

### 6c. 1D convolutional neural network (PyTorch, on `raw`)
- **Different philosophy:** instead of 561 hand-made features, the network *learns* its own
  features from the raw signals.
- **Convolution in 1D:** a small filter slides along time and responds to a local pattern
  (e.g. the shape of a footstep). Stacking layers lets later filters see longer patterns
  (the **receptive field** grows).
- **Pooling:** downsamples over time; **global average pooling** at the end summarizes the
  whole window into one vector, regardless of where in the window the pattern occurred.
- **Batch normalization:** stabilizes training. **Dropout:** randomly zeroes units during
  training to reduce overfitting.
- **Normalize each channel** using mean/std computed on the *training* fold only.
- **Early stopping:** stop when validation loss stops improving. Make the validation set
  out of *held-out subjects*, not random windows (see section 3).
- **Loss:** cross-entropy. **Optimizer:** Adam/AdamW is a sensible default.

**Check yourself**
- Why does logistic regression need scaling but LightGBM doesn't?
- What does a 1D convolution filter "look for" in an accelerometer signal?
- If the CNN matches LightGBM's accuracy, which would you ship on a watch, and why? (Think model size, battery, latency, maintainability.)

---

## 7. Tune hyperparameters without cheating

- Never tune on the test set. Tune inside the training data using grouped CV, or on a
  validation split of held-out subjects.
- Gold standard: **nested cross-validation** (inner loop tunes, outer loop scores).
  Worth describing even if you only run a simpler version.
- Keep a record of what you tried. Reporting only the best of 50 attempts inflates results.

**Check yourself**
- Why does picking the best of many models on the test set make the test score optimistic?

---

## 8. Error analysis: go past the scoreboard

- Expect **sitting vs standing** to be the main confusion. Both are static postures with
  little motion; the main difference is the phone's tilt relative to gravity.
- Look at misclassified windows: plot their raw signals next to correct ones.
- Ask what *data* or *feature* would fix it (e.g. a second sensor location, or
  orientation features from `total_acc`).

**Check yourself**
- Why are dynamic activities (walking variants) usually easier than static ones?
- What would you ask the hardware or product team for to fix the biggest confusion?

---

## 9. How much data do we need? (learning curve)

- Train on 5, 10, 15, 20... subjects; always test on held-out subjects.
- Plot score vs number of training people.
  - Still rising at the end: more labeled users would help.
  - Flat: more data won't help; change the model or the features.
- This answers a real business question: *how many people do we need to label data from?*

---

## 10. Confidence: when should the product stay quiet?

- Models output probabilities. You can choose to only show a label when the top
  probability is above a threshold, and show nothing otherwise.
- Plot **coverage** (fraction of windows where you make a call) vs **accuracy** on those calls.
- Check **calibration**: when the model says 80% confident, is it right about 80% of the time?

**Check yourself**
- A product manager wants zero wrong "you've been sitting" alerts. What do you change?

---

## 11. Write the recommendation

End the README with **"What I'd recommend the product team do"**, one paragraph.
For example: which model to ship and why, expected accuracy for a *new* user (with range),
the biggest weakness and how the product should handle it, and what data to collect next.

---

## 12. Interview questions this project prepares you for

1. How did you split your data, and why?
2. What is data leakage? Give an example from your project.
3. Why compare against a baseline?
4. Accuracy vs F1: when does each mislead?
5. Linear model vs trees vs neural net: when would you choose each?
6. How did you choose hyperparameters without overfitting to the test set?
7. Where does your model fail, and why?
8. How would this model behave for a user unlike anyone in your training data?
9. How would you decide whether to ship it?

---

## Suggested build order (checklist)

- [ ] Load data, check shapes, plot class counts and a few raw signals per activity
- [ ] Majority baseline under subject-grouped CV
- [ ] Logistic regression: grouped CV vs random CV (the leakage experiment)
- [ ] LightGBM under grouped CV, plus feature importances
- [ ] 1D-CNN under grouped CV, with subject-held-out early stopping
- [ ] Comparison table: accuracy and macro-F1, mean +/- std across folds
- [ ] Confusion matrix and per-subject scores for the best model
- [ ] Learning curve by number of training subjects
- [ ] Confidence threshold: coverage vs accuracy
- [ ] README results and product recommendation
