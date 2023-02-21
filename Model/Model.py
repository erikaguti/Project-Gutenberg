#%%
from bookclass import *


#%%
book_folders = listdir("/home/djtom/bse/term2/text/termpaper/Project-Gutenberg/Books")

def f(b):
    try:
        re.findall(r'\/*([^\/]*).txt',b)[0]
    except:
        print(b)
books=[Book_raw_data(b) for b in book_folders if b.endswith('.txt') ]


#%%
os.chdir('/home/djtom/bse/term2/text/termpaper/Project-Gutenberg/Books')
#%%
from tqdm import tqdm
for b in tqdm(books):
    b.load_all_combined()
os.chdir(os.path.dirname(os.path.realpath(__file__)))

#%%
import numpy as np
big_matrix = np.ones([len(books),200])
# big_matrix_mean0 = np.ones(big_matrix.shape)
stop_val = 1.0
for i,b in enumerate(books):
    if i%100 == 0:
        print(i)
    # print(b.title)
    a = b.chopper_sliding(my_LabMT,num_points=200,stop_val=stop_val)
    big_matrix[i,:] = b.timeseries
print(big_matrix.shape)
#%%
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2, weights="uniform")
big_matrix=imputer.fit_transform(big_matrix)

#%%
big_matrix_mean0 = big_matrix-np.tile(big_matrix.mean(axis=1),(200,1)).transpose()
import matplotlib.pyplot as plt
print(big_matrix[0,:])
fig = plt.figure()
ax = fig.add_axes([.2,.2,.7,.7])
ax.plot(big_matrix[0,:])
ax.set_xlabel("Time")
ax.set_ylabel("Happs")
ax.set_title("Example time series: {}".format(books[0].title))
# mysavefig("example-timeseries.pdf",folder="media/figures/SVD",openfig=False)
#%%
from sklearn import metrics
from sklearn.cluster import KMeans
# from the demo
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.preprocessing import scale
#%%
np.where(np.isnan(big_matrix)==True)
#%%
big_matrix[1,1]
books[17].timeseries
#%%
pca = PCA(n_components=25)
pca.fit(big_matrix)
#%%
fig = plt.figure(figsize=(7.5,5))
ax1 = fig.add_axes([0.2,0.2,0.7,0.7])
ax1.plot(pca.explained_variance_ratio_,color=".1",linewidth=2)
ax1.set_ylabel('explained variance ratio',fontsize=14)
ax1.set_xlabel('components',fontsize=14)
# mysavefig("PCA-ncomponents-variance.pdf",folder="media/figures/SVD",openfig=False)


#%%
fig = plt.figure(figsize=(15,20))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.plot(pca.components_[i],color=".1",linewidth=1.5)
    plt.ylim([-.15,.15])
plt.subplot(4,3,2)
plt.title("PCA Components for {} books, unweighted\n".format(len(books)),fontsize=20)
# mysavefig('pca-MLEcomponents-first12.png')
# mysavefig("PCA-ncomponents-top12-timeseries-weighted.pdf",folder="media/figures/SVD",openfig=False)
#%%
fig = plt.figure(figsize=(15,20))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.plot(pca.components_[i]*pca.explained_variance_ratio_[i],color=".1",linewidth=1.5)
    plt.ylim([-.01,.06])
plt.subplot(4,3,2)
plt.title("PCA Components for {} books, weighted\n".format(len(books)),fontsize=20)
# mysavefig('pca-MLEcomponents-first12.png')
# mysavefig("PCA-ncomponents-top12-timeseries-unweighted.pdf",folder="media/figures/SVD",openfig=False)
#%%
pca = PCA(n_components=12)
pca.fit(big_matrix_mean0)
print(pca.n_components_)
#%%
fig = plt.figure(figsize=(7.5,5))
ax1 = fig.add_axes([0.2,0.2,0.7,0.7])
ax1.plot(pca.explained_variance_ratio_,linewidth=2,color=".1")
ax1.set_ylabel('explained variance ratio',fontsize=14)
ax1.set_xlabel('components',fontsize=14)
# mysavefig('pca-{0}components-explainedvariance-mean0.png'.format(pca.n_components_))
# mysavefig("PCA-ncomponents-variance-mean0.pdf",folder="media/figures/SVD",openfig=False)

fig = plt.figure(figsize=(7.5,5))
ax1 = fig.add_axes([0.2,0.2,0.7,0.7])
ax1.plot(np.log10(pca.explained_variance_ratio_),color=".1",linewidth=2)
ax1.set_ylabel('log10(explained variance ratio)',fontsize=14)
ax1.set_xlabel('components',fontsize=14)
# mysavefig('pca-{0}components-explainedvariance-mean0.png'.format(pca.n_components_))
# mysavefig("PCA-ncomponents-log10variance-mean0.pdf",folder="media/figures/SVD",openfig=False)

fig = plt.figure(figsize=(15,20))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.plot(pca.components_[i],color=".1",linewidth=1.5)
    plt.ylim([-.15,.15])
plt.subplot(4,3,2)
plt.title("PCA Components for {} books, unweighted\n".format(len(books)),fontsize=20)
# mysavefig("PCA-ncomponents-timeseries-unweighted-mean0.pdf",folder="media/figures/SVD",openfig=False)
#%%

fig = plt.figure(figsize=(15,20))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.plot(pca.components_[i]*pca.explained_variance_ratio_[i],color=".1",linewidth=1.5)
    plt.ylim([-.03,.03])
plt.subplot(4,3,2)
plt.title("PCA Components for {} books, weighted\n".format(len(books)),fontsize=20)
# mysavefig('pca-MLEcomponents-first12.png')
# mysavefig("PCA-ncomponents-timeseries-weighted-mean0.pdf",folder="media/figures/SVD",openfig=False)

#%%
# pca = PCA(n_components='mle')
svd = TruncatedSVD(n_components=12,algorithm='arpack')
svd.fit(big_matrix)
# svd.n_components_

fig = plt.figure(figsize=(7.5,5))
ax1 = fig.add_axes([0.2,0.2,0.7,0.7])
ax1.plot(np.log10(svd.explained_variance_ratio_),linewidth=2,color=".1")


# pca = PCA(n_components='mle')
svd2 = TruncatedSVD(n_components=12,algorithm='arpack')
svd2.fit(big_matrix_mean0)
# svd.n_components_

ax1.plot(np.log10(svd2.explained_variance_ratio_),linewidth=2,color=".4")
ax1.legend(['SVD','SVD Mean 0'])
ax1.set_ylabel('log10(explained variance ratio)',fontsize=14)
ax1.set_xlabel('components',fontsize=14)
# mysavefig("SVD-variance.pdf",folder="media/figures/SVD",openfig=False)

fig = plt.figure(figsize=(7.5,5))
ax1 = fig.add_axes([0.2,0.2,0.7,0.7])
ax1.plot(svd.explained_variance_ratio_,linewidth=2,color=".1")
ax1.plot(svd2.explained_variance_ratio_,linewidth=2,color=".4")
ax1.legend(['SVD','SVD Mean 0'])
ax1.set_ylabel('explained variance ratio',fontsize=14)
ax1.set_xlabel('components',fontsize=14)
# mysavefig('svd-{0}components-30-50-explainedvariance-both.svg'.format(12))
# mysavefig('svd-{0}components-30-50-explainedvariance-both.png'.format(12))
# mysavefig("SVD-log10variance.pdf",folder="media/figures/SVD",openfig=False)

#%%
fig = plt.figure(figsize=(15,20))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.plot(svd.components_[i]*svd.explained_variance_ratio_[i],color=".1",linewidth=1.5)
    plt.ylim([-.02,.06])
plt.subplot(4,3,2)
plt.title("SVD Components for {} books, weighted\n".format(len(books)),fontsize=20)
# mysavefig('pca-MLEcomponents-first12.png')
# mysavefig("SVD-timeseries-weighted.pdf",folder="media/figures/SVD",openfig=False)

fig = plt.figure(figsize=(15,20))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.plot(svd.components_[i],color=".1",linewidth=1.5)
    plt.ylim([-.15,.15])
plt.subplot(4,3,2)
plt.title("SVD Components for {} books, unweighted\n".format(len(books)),fontsize=20)
# mysavefig('pca-MLEcomponents-first12.png')
# mysavefig("SVD-timeseries-unweighted.pdf",folder="media/figures/SVD",openfig=False)
#%%
fig = plt.figure(figsize=(15,20))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.plot(svd2.components_[i]*svd2.explained_variance_ratio_[i],color=".1",linewidth=1.5)
    plt.ylim([-.03,.035])
plt.subplot(4,3,2)
plt.title("SVD Mean 0 Components for {} books, weighted".format(len(books)),fontsize=20)
# mysavefig('pca-MLEcomponents-first12.png')
# mysavefig("SVD-timeseries-weighted-mean0.pdf",folder="media/figures/SVD",openfig=False)

fig = plt.figure(figsize=(15,20))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.plot(svd2.components_[i],color=".1",linewidth=1.5)
    plt.ylim([-.15,.15])
plt.subplot(4,3,2)
plt.title("SVD Mean 0 Components for {} books, unweighted".format(len(books)),fontsize=20)
# mysavefig('pca-MLEcomponents-first12.png')
# mysavefig("SVD-timeseries-unweighted-mean0.pdf",folder="media/figures/SVD",openfig=False)

#%%
def mode_plot_tight(title,modes,submodes,saveas,ylim=.15):
    num_x = 3
    num_y = len(modes)/num_x
    xspacing = .01
    yspacing = .01
    xoffset = .07
    yoffset = .07
    xwidth = (1.-xoffset)/(num_x)-xspacing
    yheight = (1.-yoffset)/(num_y)-yspacing
    print('xwidth is {0}'.format(xwidth))
    print('yheight is {0}'.format(yheight))

    fig = plt.figure(figsize=(7.5,10))
    for i,mode in enumerate(modes):
#         print(i)
#         print("====")
#         print((i-i%num_x))
        # ind = np.argsort(w[:,sv+svstart])[-20:]
        ax1rect = [xoffset+(i%num_x)*(xspacing+xwidth),1.-yheight-yspacing-(int(np.floor((i-i%num_x)/num_x))*(yspacing+yheight)),xwidth,yheight]
        ax1 = fig.add_axes(ax1rect)
        # plt.subplot(4,3,i+1)
        ax1.plot(submodes[i],color=".4",linewidth=1.5)
        ax1.plot(modes[i],color=".1",linewidth=1.5)
        ax1.set_ylim([-ylim,ylim])
        if not i%num_x == 0:
            ax1.set_yticklabels([])
            if int(np.floor((i-i%num_x)/num_x)) == num_y-1:
                ax1.set_xticks([50,100,150,200])
        if not int(np.floor((i-i%num_x)/num_x)) == num_y-1:
            ax1.set_xticklabels([])
#         if int(np.floor((i-i%num_x)/num_x)) == num_y-1 and i%num_x == 1:
#             ax1.set_xlabel("Time")
#         if i == 0:
#             new_ticks = [x for x in ax1.yaxis.get_ticklocs()]
#             ax1.set_yticks(new_ticks)
#             new_ticklabels = [str(x) for x in new_ticks]
#             new_ticklabels[-1] = "Happs"
#             # ax1.set_yticklabels(new_ticklabels)
        props = dict(boxstyle='square', facecolor='white', alpha=1.0)
        # fig.text(ax1rect[0]+.03/xwidth, ax1rect[1]+ax1rect[3]-.03/yheight, letters[i],
        my_ylim = [-ylim,ylim]
        ax1.text(.035*200, my_ylim[0]+.965*(my_ylim[1]-my_ylim[0]), "{0}".format(i),
                     fontsize=14,
                     verticalalignment='top',
                     horizontalalignment='left',
                     bbox=props)
        if i%num_x == 0:
            # new_ticks = [x for x in ax1.yaxis.get_ticklocs()]
            # ax1.set_yticks(new_ticks[:-2])
            ax1.set_yticks([-.1,0,.1])
    fig.text((1.-xoffset)/2.+xoffset,yoffset/2.,"Percentage of book",verticalalignment='center', horizontalalignment='center',fontsize=15) #,horizontalalign="center")    
    # plt.subplot(4,3,2)
    fig.text(0,(1.-yoffset)/2.+yoffset,r"h_{\textnormal{avg}}",verticalalignment='center', horizontalalignment='center',fontsize=15,rotation=90) #,horizontalalign="center"
    
    # mysavefig('pca-MLEcomponents-first12.png')
    # mysavefig(saveas,folder="media/figures/SVD",openfig=False)
    
weighted = [svd2.components_[i]*svd2.explained_variance_ratio_[i]/svd2.explained_variance_ratio_[0] for i in range(12)]
mode_plot_tight("SVD Mean 0 Components for {} books, unweighted".format(len(books)),svd2.components_,weighted,"SVD-timeseries-unweighted-mean0.pdf")

#%%
allMax = np.amax(big_matrix,axis=1)
allMin = np.amin(big_matrix,axis=1)
plt.hist(allMax,bins=100,alpha=0.7)
plt.hist(allMin,bins=100,alpha=0.7)
#%%
U,S,V = np.linalg.svd(big_matrix_mean0,full_matrices=True,compute_uv=True)
fig = plt.figure(figsize=(7.5,5))
ax1 = fig.add_axes([0.2,0.2,0.7,0.7])
ax1.plot(S,linewidth=2,color=".1")
ax1.set_ylabel('singular values Sigma',fontsize=14)
ax1.set_xlabel('components',fontsize=14)
# mysavefig('pca-{0}components-explainedvariance-mean0.png'.format(pca.n_components_))
# mysavefig("SVD-variance-numpy.pdf",folder="media/figures/SVD",openfig=False)

fig = plt.figure(figsize=(7.5,5))
ax1 = fig.add_axes([0.2,0.2,0.7,0.7])
ax1.plot(np.log10(S[:-1]),color=".1",linewidth=2)
ax1.set_ylabel('log_10(Sigma)',fontsize=14)
ax1.set_xlabel('components',fontsize=14)
# mysavefig('pca-{0}components-explainedvariance-mean0.png'.format(pca.n_components_))
# mysavefig("SVD-log10variance-numpy.pdf",folder="media/figures/SVD",openfig=False)
#%%
print(U.shape)
print(S.shape)
print(V.shape)
#%%
fig = plt.figure(figsize=(15,20))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.plot(V[i,:],color=".1",linewidth=1.5)
    plt.ylim([-.15,.15])
plt.subplot(4,3,2)
plt.title("SVD Mean 0 Components for {} books, unweighted\n".format(len(books)),fontsize=20)
# mysavefig('pca-MLEcomponents-first12.png')
# mysavefig("SVD-timeseries-unweighted-mean0-numpy.pdf",folder="media/figures/SVD",openfig=False)

fig = plt.figure(figsize=(15,20))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.plot(V[i,:]*S[i],color=".1",linewidth=1.5)
    plt.ylim([-3,3])
plt.subplot(4,3,2)
plt.title("SVD Mean 0 Components for {} books, weighted\n".format(len(books)),fontsize=20)
# mysavefig("SVD-timeseries-weighted-mean0-numpy.pdf",folder="media/figures/SVD",openfig=False)

#%%
#using the usv to examine mode contributions
# print(U[0,0]*S[0])
# print(U[0,:200]*S)
w = U[:,:200]*S
# each row entry of w are the contribution of each mode to the timeseries for book i
# where all of book i's entries are in row i
# so, the contribution from mode 1 to all books is column 1
print(w.shape)

i = 5
print(w[i,:].sum())
print(np.abs(w[i,:]).sum())
plt.figure(figsize=(15,5))
plt.plot(w[i,:],".",color=".1",linewidth=1,markersize=8)
plt.xlim([-1,200])
plt.title('contribution W of unweighted modes V to "{0}"'.format(books[i].title))
plt.xlabel('mode')
plt.ylabel('contribution')
# mysavefig("SVD-coeff-W-unweighted.pdf",folder="media/figures/SVD",openfig=False)

plt.figure(figsize=(15,5))
plt.plot(np.abs(w[i,:]),".",color=".1",linewidth=1,markersize=8)
plt.xlim([-1,200])
plt.title('contribution W of unweighted modes V to "{0}"'.format(books[i].title))
plt.xlabel('mode')
plt.ylabel('abs contribution')
# mysavefig("SVD-coeff-W-unweighted-abs.pdf",folder="media/figures/SVD",openfig=False)

plt.figure(figsize=(15,5))
plt.plot(np.cumsum(w[i,:]),".-",color=".1",linewidth=2,markersize=8)
plt.xlim([-1,200])
plt.title('contribution W of unweighted modes V to "{0}"'.format(books[i].title))
plt.xlabel('mode')
plt.ylabel('cum contribution')
# mysavefig("SVD-coeff-W-unweighted-cumsum.pdf",folder="media/figures/SVD",openfig=False)

plt.figure(figsize=(15,5))
plt.plot(np.cumsum(np.abs(w[i,:])),".-",color=".1",linewidth=2,markersize=8)
plt.xlim([-1,200])
plt.title('contribution W of unweighted modes V to "{0}"'.format(books[i].title))
plt.xlabel('mode')
plt.ylabel('cum abs contribution')
# mysavefig("SVD-coeff-W-unweighted-abs-cumsum.pdf",folder="media/figures/SVD",openfig=False)


plt.figure(figsize=(15,5))
plt.plot(U[i,:200],".",color=".1",linewidth=1,markersize=8)
plt.xlim([-1,200])
plt.title('contribution W of weighted modes V to "{0}"'.format(books[i].title))
plt.xlabel('mode')
plt.ylabel('unweigted contribution')
# mysavefig("SVD-coeff-W-weighted.pdf",folder="media/figures/SVD",openfig=False)

plt.figure(figsize=(15,5))
plt.plot(np.abs(U[i,:200]),".",color=".1",linewidth=1,markersize=8)
plt.xlim([-1,200])
plt.title('contribution W of weighted modes V to "{0}"'.format(books[i].title))
plt.xlabel('mode')
plt.ylabel('abs unweigted contribution')
# mysavefig("SVD-coeff-W-weighted-abs.pdf",folder="media/figures/SVD",openfig=False)

plt.figure(figsize=(15,5))
plt.plot(np.cumsum(U[i,:200]),".-",color=".1",linewidth=1,markersize=8)
plt.xlim([-1,200])
plt.title('contribution W of weighted modes V to "{0}"'.format(books[i].title))
plt.xlabel('mode')
plt.ylabel('cum unweigted contribution')
# mysavefig("SVD-coeff-W-weighted-cumsum.pdf",folder="media/figures/SVD",openfig=False)

plt.figure(figsize=(15,5))
plt.plot(np.cumsum(np.abs(U[i,:200])),".-",color=".1",linewidth=1,markersize=8)
plt.xlim([-1,200])
plt.title('contribution W of weighted modes V to "{0}"'.format(books[i].title))
plt.xlabel('mode')
plt.ylabel('cum abs unweigted contribution')
# mysavefig("SVD-coeff-W-weighted-abs-cumsum.pdf",folder="media/figures/SVD",openfig=False)

#%%
np.abs(w[:10,:]).sum(axis=1)
#%%
# squeeze w into the right shape
# transpose doesn't really do it
t = np.dot(np.reshape(w[0,:],(1,200)),V)
print(np.reshape(w[0,:],(1,200)).shape)
print(V.shape)
print(t.shape)
# squeeze w into the right shape
# transpose doesn't really do it
t = np.dot(w,V)
print(V.shape)
print(t.shape)