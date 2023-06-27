# gridcv_lightgbm
=======

Use LightGBM to build machine learning model, and then use grid search for optimization

## 网格搜索调参-基于LightGBM算法分类器
=====================================================================


GridSearchCV官方网址：
*<https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV>*

LightGBM官方网址：
*<https://lightgbm.readthedocs.io/en/v3.3.2/>*

#### **1.原理解释**

GridSearchCV，它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数。但是这个方法适合于小数据集，一旦数据的量级上去了，很难得出结果。这个时候就是需要动脑筋了。数据量比较大的时候可以使用一个快速调优的方法——坐标下降。它其实是一种贪心算法：拿当前对模型影响最大的参数调优，直到最优化；再拿下一个影响最大的参数调优，如此下去，直到所有的参数调整完毕。这个方法的缺点就是可能会调到局部最优而不是全局最优，但是省时间省力，巨大的优势面前，还是试一试吧，后续可以再拿bagging再优化。

通常算法不够好，需要调试参数时必不可少。比如SVM的惩罚因子C，核函数kernel，gamma参数等，对于不同的数据使用不同的参数，结果效果可能差1-5个点，sklearn为我们提供专门调试参数的函数grid_search。

*class sklearn.model_selection.GridSearchCV(estimator, param_grid, , scoring=None, n_jobs=None, iid=‘deprecated’, refit=True, cv=None, verbose=0, pre_dispatch='2n_jobs’, error_score=nan, return_train_score=False)[source]*

#### **2.Parameters**

(1)estimator：estimator object.
选择使用的分类器，并且传入除需要确定最佳的参数之外的其他参数。每一个分类器都需要一个scoring参数，或者score方法：estimator=RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features=‘sqrt’,random_state=10),

(2)param_grid：dict or list of dictionaries
需要最优化的参数的取值，值为字典或者列表，例如：param_grid =param_test1，param_test1 = {‘n_estimators’:range(10,71,10)}。

(3)scoring：str, callable, list/tuple or dict, default=None
模型评价标准，默认None,这时需要使用score函数；或者如scoring=‘roc_auc’，根据所选模型不同，评价准则不同。字符串（函数名），或是可调用对象，需要其函数签名形如：scorer(estimator, X, y)；如果是None，则使用estimator的误差估计函数。具体取值可访问*https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter*

(4)n_jobs：int, default=None
n_jobs: 并行数，int：个数,-1：跟CPU核数一致, 1:默认值

(5)pre_dispatch：int, or str, default=n_jobs
指定总共分发的并行任务数。当n_jobs大于1时，数据将在每个运行点进行复制，这可能导致OOM，而设置pre_dispatch参数，则可以预先划分总共的job数量，使数据最多被复制pre_dispatch次

(6)iid：bool, default=False
iid:默认True,为True时，默认为各个样本fold概率分布一致，误差估计为所有样本之和，而非各个fold的平均。

(7)cvint, cross-validation generator or an iterable, default=None
交叉验证参数，默认None，使用三折交叉验证。指定fold数量，默认为3，也可以是yield训练/测试数据的生成器。

(8)refit：bool, str, or callable, default=True
默认为True,程序将会以交叉验证训练集得到的最佳参数，重新对所有可用的训练集与开发集进行，作为最终用于性能评估的最佳模型参数。即在搜索参数结束后，用最佳参数结果再次fit一遍全部数据集。

(9)verbose：integer
verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。

(10)error_score：‘raise’ or numeric, default=np.nan
如果估算器拟合出现错误，则分配给分数的值。 如果设置为“ raise”，则会引发错误。 如果给出数值，则引发FitFailedWarning。 此参数不会影响重新安装步骤，这将始终引发错误。

(11)return_train_score：bool, default=False
如果“False”，cv_results_属性将不包括训练分数

#### **3.Attributes**

(1)cv_results_ : dict of numpy (masked) ndarrays
具有键作为列标题和值作为列的dict，可以导入到DataFrame中。注意，“params”键用于存储所有参数候选项的参数设置列表。

(2)best_estimator_ : estimator
通过搜索选择的估计器，即在左侧数据上给出最高分数（或指定的最小损失）的估计器。 如果refit = False，则不可用。

(3)best_score_ : float
best_estimator的分数

(4)best_params_ : dict
在保存数据上给出最佳结果的参数设置

(5)best_index_ : int 对应于最佳候选参数设置的索引（cv_results_数组）。
与最佳候选参数设置相对应的（cv_results_数组的）索引。
search.cv_results _ [‘params’] [search.best_index_]上的字典给出了最佳模型的参数设置，该模型给出了最高的平均分数（search.best_score_）。
对于多指标评估，仅当指定重新安装时才存在。

(6)scorer_ : function or a dict
在保留的数据上使用记分器功能，以为模型选择最佳参数。
对于多指标评估，此属性包含将得分者键映射到可调用的得分者的有效得分dict。

(7)n_splits_ : int
交叉验证拆分的数量（折叠/迭代）。

(8)refit_time_：float
用于在整个数据集中重新拟合最佳模型的秒数。仅当改装不为False时才存在。0.20版中的新功能。



#### **4.调优步骤**

（1）设置参数的初始值，简单看下效果

（2）最大深度和叶子数，先粗调，再细调
sklearn模型评估里的scoring参数都是采用的higher return values are better than lower return values（较高的返回值优于较低的返回值）。但是，我采用的metric策略采用的是auc，所以是越高越好。

（3）降低过拟合
为了将模型训练的更好，极有可能将 max_depth 设置过深或 num_leaves设置过小，造成过拟合，因此需要 min_data_in_leaf和 min_sum_hessian_in_leaf来降低过拟合。

（4）降低过拟合—两个抽样参数
feature_fraction参数来进行特征的子抽样。这个参数可以用来防止过拟合及提高训练速度。
bagging_fraction+bagging_freq参数必须同时设置，bagging_fraction相当于subsample样本采样，可以使bagging更快的运行，同时也可以降拟合。bagging_freq默认0，表示bagging的频率，0意味着没有使用bagging，k意味着每k轮迭代进行一次bagging。


（5）降低过拟合—正则项
正则化参数lambda_l1(reg_alpha), lambda_l2(reg_lambda)，毫无疑问，是降低过拟合的，两者分别对应l1正则化和l2正则化。我们也来尝试一下使用这两个参数。

（6）降低学习率
由于之前使用了较高的学习速率是可以让收敛更快，但是准确度不够，一次使用较低的学习速率，以及使用更多的决策树n_estimators来训练数据，看能不能可以进一步的优化分数。同时我们可以用回lightGBM的cv函数 ，代入之前优化好的参数看结果。

好了，以上就是网络搜素调优的一般步骤，本文仅是对LightGBM来说一些重要的参数进行调优，也可以对其他的参数进行调优，具体看自己的需求。

