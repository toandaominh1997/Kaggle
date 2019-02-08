# Kaggle

Need info? Check the [Wiki](https://github.com/toandaominh1997/kaggle/wiki)
 | or [Create an issue](https://github.com/toandaominh1997/kaggle/issues/new)
 | Check [our project Board](https://github.com/toandaominh1997/kaggle/projects)
 | [Ask us on Discord][discord] | Support us on [Toandaominh1997](https://github.com/toandaominh1997)

[![chat][chat-badge]][chat]
[![Build Status][build-badge]][build]
[![version][version-badge]][package]
[![MIT License][license-badge]][LICENSE]

[![All Contributors](https://img.shields.io/badge/all_contributors-18-orange.svg?style=flat-square)](#contributors)
[![PRs Welcome][prs-badge]][prs]
[![Implementations][implementations-badge]][implementations]
[![Donate][donate-badge]][donate]
[![Code of Conduct][coc-badge]][coc]

[![Watch on GitHub][github-watch-badge]][github-watch]
[![Star on GitHub][github-star-badge]][github-star]
[![Tweet][twitter-badge]][twitter]


---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Template Code](#template-code)
- [Feedback](#feedback)
- [Contributors](#contributors)
- [Build Process](#build-process)
- [Backers](#backers-)
- [Sponsors](#sponsors-)
- [Acknowledgments](#acknowledgments)
- [Contributors](#contributors)
- [License](#license)



## Introduction


## Features

## Fun with Kaggle
### 1. Memory saving function for pandas
```python
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

```
### 2. Plotting Learning Curves

```python
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

```
### 3. Cross Validation
```python
def cross_validation(train, target, test, model, folds, N_FOLDS=10):
    oof = np.zeros(len(train))
    sub = np.zeros(len(test))
    scores = [0 for _ in range(folds.n_splits)]
    for fold_, (train_index, val_index) in enumerate(folds.split(X, y)):
        if(isinstance(X,(pd.core.frame.DataFrame))):
            X_train, y_train = train.loc[train_index], y.loc[train_index]
            X_val, y_val = train.loc[val_index], y.loc[val_index]
        model.fit(X_train, y_train)
        oof[val_idx] = model.predict(X_val)
        sub += model.predict(test)/folds.n_splits
        scores[fold_] = roc_auc_score(y[val_index], oof[val_index])
        print("Fold {}: {}".format(fold_+1, round(scores[fold_],5)))
    print("CV score(auc): {:<8.5f}, (std: {:<8.5f})".format(roc_auc_score(y, oof), np.std(scores)))
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
clf = LogisticRegression(C=13.0)
cross_validation(train=X, target=y, test=test, model=clf, folds=folds)    
```
## Template Code

### LightGBM

``` python
import lightgbm as lgb


params = {'learning_rate': 0.2,
              'application': 'binary',
              'num_leaves': 31,
              'verbosity': -1,
              'metric': 'auc',
              'data_random_seed': 2,
              'bagging_fraction': 0.8,
              'feature_fraction': 0.6,
              'nthread': 4,
              'lambda_l1': 1,
              'lambda_l2': 1}
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)
watchlist = [train_data, val_data]

model_lgb = lgb.train(params, train_set=d_train, valid_sets=watchlist)

y_pred = model_lgb.predict(X_test)
```


## Feedback

Feel free to send us feedback on [Twitter](https://twitter.com/gitpointapp) or [file an issue](https://github.com/gitpoint/git-point/issues/new). Feature requests are always welcome. If you wish to contribute, please take a quick look at the [guidelines](./CONTRIBUTING.md)!

If there's anything you'd like to chat about, please feel free to join our [Gitter chat](https://gitter.im/git-point)!

## Contributors

This project follows the [all-contributors](https://github.com/kentcdodds/all-contributors) specification and is brought to you by these [awesome contributors](./CONTRIBUTORS.md).

## License

Github Changelog Generator is released under the [MIT License](http://www.opensource.org/licenses/MIT).


[chat-badge]: https://img.shields.io/badge/chat-on%20gitter-46BC99.svg?style=flat-square
[chat]: https://gitter.im/kentcdodds/all-contributors?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
[build-badge]: https://img.shields.io/travis/kentcdodds/all-contributors.svg?style=flat-square
[build]: https://travis-ci.org/kentcdodds/all-contributors
[version-badge]: https://img.shields.io/npm/v/all-contributors.svg?style=flat-square
[package]: https://www.npmjs.com/package/all-contributors
[license-badge]: https://img.shields.io/npm/l/all-contributors.svg?style=flat-square
[license]: https://github.com/kentcdodds/all-contributors/blob/master/LICENSE
[prs-badge]: https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square
[prs]: http://makeapullrequest.com
[donate-badge]: https://img.shields.io/badge/$-support-green.svg?style=flat-square
[donate]: https://kcd.im/donate
[coc-badge]: https://img.shields.io/badge/code%20of-conduct-ff69b4.svg?style=flat-square
[coc]: https://github.com/kentcdodds/all-contributors/blob/master/other/CODE_OF_CONDUCT.md
[implementations-badge]: https://img.shields.io/badge/%F0%9F%92%A1-implementations-8C8E93.svg?style=flat-square
[implementations]: https://github.com/kentcdodds/all-contributors/blob/master/other/IMPLEMENTATIONS.md
[github-watch-badge]: https://img.shields.io/github/watchers/toandaominh1997/Kaggle.svg?style=social
[github-watch]: https://github.com/toandaominh1997/Kaggle/watchers
[github-star-badge]: https://img.shields.io/github/stars/toandaominh1997/Kaggle.svg?style=social
[github-star]: https://github.com/toandaominh1997/Kaggle/stargazers
[twitter]: https://twitter.com/intent/tweet?text=Check%20out%20all-contributors!%20%E2%9C%A8%20Recognize%20all%20contributors,%20not%20just%20the%20ones%20who%20commit%20code%20%E2%9C%A8%20https://github.com/toandaominh1997/Kaggle%20%F0%9F%A4%97
[twitter-badge]: https://img.shields.io/twitter/url/https/github.com/kentcdodds/all-contributors.svg?style=social
[emojis]: https://github.com/toandaominh1997/Kaggle/#emoji-key
[all-contributors]: https://github.com/toandaominh1997/Kaggle/

