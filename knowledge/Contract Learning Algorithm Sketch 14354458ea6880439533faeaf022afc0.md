# Contract Learning Algorithm Sketch

Author: Grigorii Kirgizov

This is a development of the approach described in [Idea on ARC Contract Learning](https://app.notion.com/p/Idea-on-ARC-Contract-Learning-10554458ea688064958ff94b3eb31da2?pvs=21) 

## Core principles & competitive advantages

- Any step forward is *positively justified*. This is an inductive bias of the algorithm and of the ARC domain. There's minimal number of random choices in the search space. In other words, our search is interpretable. Brute-force solutions can randomly solve the task given enough time and rich enough means. But there're only few (or only one) justified paths where each step has a principle or constraint or metric that *tells why it's chosen*.
- *Motivated observation.* We never look for all possible relations between all objects on the scene. We extract them when we understand that we need more constraints or when "something is missing", when we have insufficient information to make the inference using present information. We look for relations only between pre-identified "likely related" objects. Compare this with how detectives work. They fallback to brute-force enumeration of possibilities only when there're no more targeted choices. We should build an ARC detective, in a sense.
- We work with *multiple symbolic interpretations* of the raw input. What we see, in its turn, influences available transformations.

## Algorithm sketch (v.1)

Having:

- Set of algebraic data types for describing ARC entities: $\mathbb{T} = \{T\}$
- Library of basic components ("axioms"): $\mathbb{F} = \{~ f : T_1[C_1] \rightarrow T_2[C_2] ~\}$
- Contract language $\mathbb{L}_C$, an instance of Description Logics, including:
    - generating basis: $\{~ a_T: T \rightarrow C ~\}$ -- that extracts attributes from objects that can be used in contracts e.g. $height: Any \rightarrow Int$ or $color: Any \rightarrow Colors$.

Do the following:

1. $\{T_I\}, \{T_O\}$ :: Ground images into variants of initial symbolic interpretations
    - Try possible object recognizers. Select possible interpretations from those with success result.
2. $\{h: T_I \rightarrow T_O\}$ :: Input-Output pairing of symbolic interpretations.
3. $\{h: T_I [C_I] \rightarrow T_O[C_O]\}$ :: Extract basic contracts preserved by transformations. This is done simply by evaluating contract checking functions on objects (e.g. check position equality, color equality etc).
4. Determine consistent alternatives for hypothesa reification:
Def. **Consistency criteria**. Transformation $f : T_1[C_1] \rightarrow T_2[C_2]$ with hypothesis $h : T_I[C_I] \rightarrow T_O[C_O]$:
    - *is consistent* iff $T_1 \lhd T_I \& T_O \lhd T_2 \& C_1 \Rightarrow C_I \& C_O \Rightarrow C_2$ (i.e. subtyping of reification types)
    - *is weakly consistent* iff $T_1 \lhd T_I \& T_O \lhd T_2 \& C_1 \Rightarrow C_I \& C_O \nRightarrow \neg C_2$ (i.e. result of the function doesn't contradict the hypothesis goal)
    - So, if I don't have single matching transformation, then how do I proceed?
    There're several variants:
        - (a) if there's single strongly or weakly consistent alternative, then take it as a step in transformation path
        - (b) if there’re many strongly consistent alternatives => generalize them; and reify hypothesis to their generalization (lemma induction happens here)
        - (c) if no alternatives are strongly consistent, but some are weakly consistent => go to the next step; one of them would constitute a step in the transformation path.
        - (d) if no alternatives are weakly consistent => we're in the cul-de-sac. Finish this unsuccesful branch, explore another.
5. Invent selection condition for selecting alternatives:
    - ILP Subtask. Construct transformation application condition: $Sel_h(f) = g_h(C_I) \sim g_f(C_1), ~\sim \in \{=,\neq\};~~ g_f,g_h \in \mathbb{L}_C$
    - such that, informally, the set of alternatives is filtered until one or few alternatives are left, yet the selection condition is general enough (simple enough). This is the objective. Complexity of the condition can be estimated with Stitch-like heuristic metrics.
    - Possibly, due to narrow search domain, it can be enough to run exhaustive search (with term preference weighted by complexity).
    - The comparing relation is not necessarily an equality. More generally, it must be *selective*: it either filters or orders alternatives. Also, it can be conjunction or disjunction of conditions.
    - Commentary. This *selector invention* relies on inductive bias of ARC: every choice should be positively justified. Here invention of a selection condition *based on attributes* is the justification of choosing particular transformation. This plays in contrast to most other approaches that's based on enumeration-evaluation of possibilities without justifications. This approach is more explainable.
    
    Some standing questions:
    
    - [ ]  How to use output types & contracts $T_O[C_O], T_2[C_2]$ in this case? They provide additional information and intuitively they could influence preference over alternatives.
    - [ ]  How does availability of multiple examples can be used here? We have single hypothesis (already generalized over examples), and single set of abstract alternatives. But when applied, we get different concrete instantiations of $T_2[C_2]$. How do we use them? Do we extract additional attributes from them?
    - [ ]  How does this "selection condition invention" that narrows down the search influence search completeness?
6. Reify hypothesis:
    1. If the set of selected alternatives is sufficient for definite choice (i.e. there're one or few alternatives), then apply it.
    2. If the selected set is still large OR selection procedure didn't succeed, then generate counterfactual example: generate one or several runnable programs (randomly, in general). Reify hypothesis given contracts of counterfactual examples. Commentary: this procedure must be adapted from **Popper** ILP solver.
    - Commentary: Bear in mind that we might potentially have multiple competing hypothesa. In this case take the one where selector invention succeeds. Only if no hypothesis enables us to find definite choice, then proceed to counterfactual learning.
    - Suppose we had hypothesis $h_i : T_I[C_I] \rightarrow T_O[C_O]$, and we chose transformation $f : T_A[C_A] \rightarrow T_B[C_B]$ which is consistent with hypothesis. The next reified hypothesis would be: $h_{i+1} : T_B[C_B] \rightarrow T_O[C_O]$.
    - Our *transformation path* is then reified by one step: $T_I[C_I] \rightarrow_f T_B[C_B] \rightarrow^* T_O[C_O]$. This constitutes reduction of the original problem.
- Go to step 4 and repeat until:
    - Success: we have a complete transformation path for the original hypothesis: $h_0 : T_I[C_I] \rightarrow T_O[C_O]$.
    - Failure: alternative hypothesa are exhausted or timeout is reached.

Commentary:

- What's representation of the libraries of components? *Type transition network.* That's natural representation for searching paths. As a bonus, it is a natural representation for Noeon system.
- Some kind of learnt statistics over the search tree is required. It can be simply frequentist up to epsilon.
- Resulting "paths" of selection criteria (i.e. justifications of our choices) could be a source of insights regarding "reasoning paths": sequences of things we analyze to determine our choices.