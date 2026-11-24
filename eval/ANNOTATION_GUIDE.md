# Annotation guide: is this pair a good cross-sell bundle?

Each row names two products that shoppers often buy together. Label the **relation between the two products**,
as a retail category manager would, for one decision: should this pair appear on a flyer as
"buy product 1, get product 2 at 20 % off" (or the other way round)?

Judge from the two names only. Don't look anything up. The order of the two products does not matter.

## Labels

**complement**: different kinds of product that go together. They are used or eaten together, belong to the same
recipe, task or occasion, or are parts made to be combined. Having one gives a reason to add the other.
Test: would a shopper add B *in addition to* A, because of A?
Examples: tennis racket + tennis balls; pasta + pasta sauce; coffee machine + descaler; party plates + party napkins.

**substitute**: the same kind of product. They serve the same function and differ in colour, pattern, flavour, size,
pack, brand or a near-equivalent type. A discount would shift sales from one to the other.
Test: would a shopper pick one *instead of* the other, or is the second just more of the same?
Examples: shirt in red + shirt in blue; apple juice + orange juice; large candle + small candle; whole milk + skimmed milk;
two different card games; two kinds of shopping bag that are both sold as products.

**not_a_bundle**: there is no flyer logic, for one of two reasons:
(a) at least one line is not a product for sale: postage, fees, deposit, voucher, carrier bag handed out at the till, manual adjustment;
(b) both are products, but of different kinds with no specific link in use or occasion. They co-occur only because
people shop for several things at once.
Examples: garden trowel + birthday card; lightbulbs + cat food.

## Borderline rules

1. **Same collection, different function:** matching teapot + matching teacup is a complement, because they do different jobs at one occasion.
2. **Same function, different design:** "lunch bag apple design" + "lunch bag cars design" is a substitute, even if some shoppers buy both.
3. **Parts of one whole:** dollhouse bedroom + dollhouse kitchen are a complement, because they are made to be combined.
4. **Home decor of different kinds:** a complement only if the names show a shared style, motif or collection. Otherwise not_a_bundle.
5. **Food:** a complement needs a concrete meal, recipe or use (rolls + sausage; coffee + milk). If the only link is "both are everyday groceries", use not_a_bundle.
6. **A product that also contains the other** (mug + mug with spoon): substitute.
7. A carrier or shopping bag **sold as a designed product** (jumbo bag with a pattern, lunch bag) is a product, not an operational line. Apply the other rules to it.
8. A vague category name (for example "other vegetables") is judged by what it plausibly covers.

## Output

Write one CSV per set with exactly the header `pair_id,label,confidence,reason`:

- `label`: complement | substitute | not_a_bundle
- `confidence`: 3 = clear, 2 = probably, 1 = guess
- `reason`: at most 10 words; required when confidence is below 3, otherwise leave empty

Label every row. Never skip a row, and don't change the ids.
