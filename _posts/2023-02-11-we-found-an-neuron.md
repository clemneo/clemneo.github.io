---
layout: post
title: We Found An Neuron in GPT-2
description: 
summary: 
comments: false
category: technical
---

> Written in collaboration with Joseph Miller. 
> See the discussion of this post over on [LessWrong](https://www.lesswrong.com/posts/cgqh99SHsCv3jJYDS/we-found-an-neuron-in-gpt-2).

We started out with the question: How does GPT-2 know when to use the word `an` over `a`? The choice depends on whether the word that comes after starts with a vowel or not, but GPT-2 is only capable of predicting one word at a time.

We still don’t have a full answer, but we did find a single MLP neuron in GPT-2 Large that is crucial for predicting the token " an". And we also found that the weights of this neuron correspond with the embedding of the " an" token, which led us to find other neurons that predict a specific token.

## Discovering the neuron
### Choosing the prompt
It was surprisingly hard to think of a prompt where GPT-2 would output the token `“ an”` (the leading space is part of the token) as the top prediction. In fact, we gave up with `GPT-2_small` and switched to GPT-2_large. As we’ll see later, even `GPT-2_large` systematically under-predicts `“ an”` in favor of `“ a”`. This may be because smaller language models lean on the higher frequency of a to make a best guess. The prompt we finally found that gave a high (64%) probability for “ an” was:

> _“I climbed up the pear tree and picked a pear. I climbed up the apple tree and picked”_

The first sentence was necessary to push the model towards an indefinite article — without it the model would make other predictions such as _“\[picked\] up”_.

Before we proceed, here’s a quick overview on the [transformer architecture](https://transformer-circuits.pub/2021/framework/index.html). Each attention block and MLP takes inputs and adds outputs to the residual stream.

<img src="/assets/images/anneuron/transformer-architecture.png" style="max-width:100%">

### Logit Lens
Using a technique known as [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens), we took the logits from the residual stream between each layer and plotted the difference between` logit(‘ an’)` and `logit(‘ a’)`. We found a big spike after Layer 31’s MLP.

{% include_relative plotly/logit_lens_fig.html %}

### Activation Patching by the Layer


Activation patching is a technique introduced by [Meng et. al. (2022)](https://arxiv.org/abs/2202.05262) to analyze the significance of a single layer in a transformer.  First, we saved the activation of each layer when running the original prompt through the model — the “clean activation”.

We then ran a **corrupted** prompt through the model: _“I climbed up the pear tree and picked a pear. I climbed up the **lemon** tree and picked”_. By replacing the word ‘apple’ with ‘lemon’, we induce the model to predict the token ‘ a’ instead of ‘ an’.

With the model predicting `" a"` over `" an"`, we can replace a layer’s corrupted activation with its clean activation to see how much the model shifts towards the `" an"` token, which indicates that layer’s significance to predicting `" an"`. We repeat this process over all the layers of the model.

{% include_relative plotly/patched_resid_fig.html %}
{% include_relative plotly/patched_attn_fig.html %}

We're mostly going to ignore attention for the rest of this post, but these results indicate that Layer 26 is where `" picked"` starts thinking a lot about `" apple"`, which is obviously required to predict `" an"`.

{% include_relative plotly/patched_mlp_fig.html %}

The two MLP layers that stand out are Layer 0 and Layer 31. We already know that Layer 0’s MLP is generally important for GPT-2 to function[^1] (although we're not sure why attention in Layer 0 is important). The effect of Layer 31 is more interesting. Our results suggests that Layer 31’s MLP plays a significant role in predicting the ‘ an’ token. (See [this comment](https://www.lesswrong.com/posts/cgqh99SHsCv3jJYDS/we-found-an-neuron-in-gpt-2?commentId=FLpxtfnwnMjZwXv3B#comments) if you're confused how this result fits with the logit lens above.)

## Finding 1: We can discover predictive neurons by activation patching individual neurons
Activation patching has been used to investigate transformers by the layer, but can we push this technique further and apply it to individual neurons? Since each MLP in a transformer only has one hidden layer, each neuron’s activation does not affect any other neuron in the MLP. So we should be able to patch individual neurons, because they are independent from each other in the same sense that the attention heads in a single layer are independent from each other.

We run neuron-wise activation patching for Layer 31’s MLP in a similar fashion to the layer-wise patching above. We reintroduce the clean activation of each neuron in the MLP when running the corrupted prompt through the model, and look at how much restoring each neuron contributes to the logit difference between `" a"` and `" an"`.

{% include_relative plotly/patched_neuron_fig.html %}

We see that patching Neuron 892 recovers 50% of the clean prompt's logit difference, while patching whole layer actually does worse at 49%.


## Finding 2: The activation of the "an-neuron" correlates with the " an" token being predicted.
### Neuroscope [Layer 31 Neuron 892 Maximum Activating Examples](https://neuroscope.io/gpt2-large/31/892.html)
![Neuroscope's An](/assets/images/anneuron/neuroscope_an.png)
Neuroscope is an online tool that shows the top activating examples in a large dataset for each neuron in GPT-2. When we look at Layer 31 Neuron 892, we see that the neuron maximally activates on tokens where the subsequent token is `" an"`.

But Neuroscope only shows us the top 20 most activating examples. Would there be a trend for a wider range of activations?

### Testing the neuron on a larger dataset
To check for a trend, we ran the [pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k/tree/main/data) dataset through the model. This is a diverse set of about 10 million tokens taken from [The Pile](https://pile.eleuther.ai/), split into prompts of 1,024 tokens. We plotted the proportion of `" an"` predictions across the range of neuron activations:

{% include_relative plotly/an_neuron_proportion.html %}

We see that the proportion of `" an"` predictions increases as the neuron’s activation increases, to the point where `" an"` is always the top prediction. The trend is somewhat noisy, which suggests that there might be other mechanisms in the model that also contribute towards the ‘ an’ prediction. Or maybe when the `" an"` logit increases, other logits increase at the time.

Note that the model only predicted " an" 1,500 times, even though it actually occurred 12,000 times in the dataset. No wonder it was so hard to find a good prompt!

### The neuron’s output weights have a high dot-product with the “ an” token

How does the neuron influence the model’s output? Well, the neuron’s output weights have a high dot product with the embedding for the token “ an”. We call this the **congruence** of the neuron with the token. Compared to other random tokens like `" any"` and `" had"`, the neuron’s congruence with " an" is very high:

![Congruence Illustration](/assets/images/anneuron/congruence_illustration.png)

In fact, when we calculate the neuron’s congruence with all of the tokens, there are a few clear outliers:

{% include_relative plotly/an_neuron_vs_all_tokens.html %}

It seems like the neuron basically adds the embedding of `“ an”` to the residual stream, which increases the output probability for “ an” since the unembedding step consists of taking the dot product of the final residual with each token[^extra1].

Are there other neurons that are also congruent to `“ an”`? To find out, we can calculate the congruence of all neurons with the `“ an”` token:

{% include_relative plotly/an_token_vs_all_neurons.html %}

Our neuron is way above the rest, but there are other neurons with a fairly high congruence with the `" an"` token. These other neurons could be part of the reason why the correlation between the an-neuron’s activation and the prediction of the `" an"` token isn’t perfect: there may be prompts where `" an"` is predicted, but the model uses these other neurons to do it.

If this is the case, could we use congruence to find a neuron that is perfectly correlated with a single token prediction?

## Finding 3: We can use the neurons’ output congruence to find specific neurons that predict a token
### Finding a token-associated neuron
We can try to find a neuron that is associated with a specific token by running the following search:

1. For each token, find the neuron with the highest output congruence. 
2. For each of these congruent neurons, find how much more congruent they are as compared to the next most congruent neuron for the same token.
3. Take the neuron(s) that are the most exclusively congruent.

With this search, we wanted to find neurons that were uniquely responsible for a token. Our conjecture was that with a neuron that was mostly responsible for a token, its activation would be more correlated with the token’s prediction, since any prediction of that token would “rely” on that neuron.

Let’s run the search and plot the graph of the most congruent neurons for each token:

{% include_relative plotly/top_2_neuron_diff_for_each_token_fig.html %}

With this search, we see that for tokens like “ off” and “ though”, there are neurons that stand out in their congruence. Let’s try running the “ though” neuron — Layer 28 Neuron 1921 — through the dataset and see whether we get a cleaner graph!

{% include_relative plotly/though_neuron_proportion.html %}

Woah, that is much messier than the graph for the an-neuron. What is going on?

Looking at [Neuroscope](https://neuroscope.io/gpt2-large/28/1921.html)’s data for the neuron reveals that the max activating neuron predicts both the tokens `“ though”` and `“ however”`. This complicates things — it seems that this neuron is correlated with a group of semantically similar tokens ([conjunctive adverbs](https://en.wikipedia.org/wiki/Conjunctive_adverb))[^2]

![Though Neuroscope](/assets/images/anneuron/neuroscope_though.png)

When we calculate the neuron’s congruence for all tokens, we find that the same tokens pop up as outliers:

{% include_relative plotly/though_neuron_vs_all_tokens.html %}

In our large dataset correlation graph above, instances where the neuron activates and `" however"` is predicted over `" though"` would be counted as negative examples, since " though" was not the top prediction. This could also explain some of the noise in the `" an"` correlation, where the neuron is also congruent with `"An"`, `" An"` and `"an"`[^3].

Can we find a better neuron to look at — preferably a neuron that only predicts for one token?

### Finding a cleanly associated neuron
For a neuron to be ‘cleanly associated’ with a token, their congruence with each other should be _mutually exclusive_, meaning:

1. The neuron is much more congruent with the token than any other neuron.
2. The neuron is much more congruent with the token than any other token.

(Remember, congruence is just the dot product.)

Both criteria help to simplify the relationship between the neuron and its token. If a neuron’s congruence with a token is a representation of how much it contributes to that token’s prediction, the first criteria can be seen as making sure that **only this neuron** is responsible for predicting that token, while the second criteria can be seen as making sure that this neuron is responsible for predicting **only that token**.

Our search then is as follows:

1. For each token, find the most congruent neuron.
2. For each neuron, find the most congruent token[^4].
3. Find the token-neuron pairs that are on both lists — that is, the pairs where the neuron's most congruent token is a token which is most congruent with that neuron!
4. Calculate how distinct they are by multiplying their top 2 token congruence difference with their top 2 neuron congruence difference.
5. Find the pairs with the highest mutual exclusive congruence.

{% include_relative plotly/top_mutual_exclusive_congruence_pairs.html %}

For `GPT-2_large`, Layer 33 Neuron 4142 paired with `"i"` scores the highest on this metric. Looking at Neuroscope[^5] confirms the connection:

![Neuroscope i](/assets/images/anneuron/neuroscope_i.png)

And when we plot the graph of top prediction proportion over activation for the top 5 highest scorers[^6]:

{% include_relative plotly/multiline_top_pred_proportion_fig.html %}

We see that we do indeed get a smooth correlations for each pair!

## What Does This All Mean?
Does the congruence of a neuron with a token actually measure the extent to which the neuron predicts that token? We don't know. There could be several reasons why even token-neuron pairs with high mutually exclusive congruence may not always correlate:

* The token could be also predicted by a combination of less congruent neurons
* The token could be predicted by attention heads
* Even if a neuron’s activation has a high correlation with a token’s logit, it may also indirectly correlate with other token's logits, such that the neuron activation does not correlate with the token's probability.
* There may be later layers which add the opposite direction to the residual stream, cancelling the effect of this neuron.

However, we’ve found that the token neuron pair with the highest mutually exclusive congruence (the “i” and the “i-neuron”) does in fact have a strong correlation. We haven't tested any others pairs yet but we expect that many others pairs that score high on this metric will also correlate.

## TL;DR
1. We used activation patching on a neuron level to find a neuron that's important for predicting the token `" an"` in a specific prompt.
2. The “an-neuron” activation actually correlates with `" an"` being predicted in general.
3. This may be because the neuron’s output weights have a high dot product with the `" an"` token (the neuron is highly _congruent_ with the token). Moreover this neuron has a higher dot product with this token than any other token. **And** this neuron has a higher dot product with this token than the token has with any other neuron (they have high mutual exclusive congruence).
4. The congruence between a neuron and a token is cool. We find the "i" neuron-token pair which has the highest _mutual exclusive congruence_ of any token-neuron pair. The activation of this neuron is strongly correlated with the `"i"` token being predicted.

The code to reproduce our results can be found [here](https://github.com/UFO-101/an-neuron).

This is a write-up and extension of our winning submission to [Apart Research](https://apartresearch.com/)'s [Mechanistic Interpretability Hackathon](https://itch.io/jam/mechint). Thanks to the London EA Hub for letting us use their co-working space, [Fazl Barez](https://fbarez.github.io/) for his comments and [Neel Nanda](https://www.neelnanda.io/) for his feedback and for creating [Neuroscope](https://neuroscope.io/), the [pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) dataset and [TransformerLens](https://github.com/neelnanda-io/TransformerLens).

## Footnotes
[^1]: 
    [Neel Nanda’s take on MLP 0](https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/main/Exploratory_Analysis_Demo.ipynb#scrollTo=i3NvQUOXW3bC&line=15&uniqifier=1):

    "It's often observed on GPT-2 Small that MLP0 matters a lot, and that ablating it utterly destroys performance. My current best guess is that the first MLP layer is essentially acting as an extension of the embedding (for whatever reason) and that when later layers want to access the input tokens they mostly read in the output of the first MLP layer, rather than the token embeddings. Within this frame, the first attention layer doesn't do much.

    In this framing, it makes sense that MLP0 matters on the second subject token, because that's the one position with a different input token!

    I'm not entirely sure why this happens, but I would guess that it's because the embedding and unembedding matrices in GPT-2 Small are the same. This is pretty unprincipled, as the tasks of embedding and unembedding tokens are not inverses, but this is common practice, and plausibly models want to dedicate some parameters to overcoming this.
    
    I only have suggestive evidence of this, and would love to see someone look into this properly!”

[^extra1]:
    What else could it have done? It might have suppressed the logit for " a" which would have had the same impact on the logit difference. Or it might have added some completely different direction to the residual which would cause a neuron in a later layer to increase the " an" logit.

[^2]: 
    Note that while the though-neuron is congruent to a group of semantically similar tokens, the an-neuron is correlated with a group of _syntactically_ similar tokens (eg. `" an"` and `" Ancients"`).

[^3]:
    Why does `" an"` have a cleaner correlation despite the other congruent tokens? We're not sure. One possible explanation is that `"An"` and `" An"` are simply much less common tokens so they make little impact on the correlation, while `"an"` has a significantly lower congruence with the neuron than the top 3.

    In general, we expect that neurons found by only looking at the top 2 neuron difference for each token will not often have clean correlations with their respective tokens because these neurons may be congruent with multiple tokens.


[^4]: 
    When we look at the most congruent tokens for each neuron, we see some [familiar troublemakers](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation) showing up with very high congruence:


    ![Top Neurons For Each Token](/assets/images/anneuron/top_neuron_for_each_token.png)

    At first, it looks like these 'forbidden tokens' are all associated with a 'forbidden neuron' (Layer 35 Neuron 3354) which they are all very congruent with. But actually if we plot the most congruent tokens of many other neurons we also see some of these weird tokens showing up. Our tentative hypothesis is that this has something to do with the [hubness effect](https://www.lesswrong.com/posts/Ya9LzwEbfaAMY8ABo/?commentId=M2uAwsCus2adqQsGc).

[^5]:
    Neuroscope data wasn't available for this neuron, so we took the max activating dataset examples from the pile-10k dataset. Texts 1, 2, 3 are examples 1755, 8528 and 6375 respectively. 

[^6]:
    Note that one of the top 5 tokens is `'an'`, but this is quite different from `' an'` that we were talking about earlier as it will rarely be used as the start of a word or a word on its own. And similarly the neuron with which it is paired, Layer 34 Neuron 4549, is not the an-neuron we were talking about earlier.