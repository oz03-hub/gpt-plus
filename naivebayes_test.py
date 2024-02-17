from naivebayes import NaiveBayes, depickle_data

nb = NaiveBayes(["image", "text"], "./datasets")
nb.train("./pickle-db/pickle")
print(nb.evaluate_classifier())

print(
    nb.classify(
        nb.tokenize(
            "Generate a list of authors who have written a famous book that is translated in over 10 languages"
        )
    )
)

print(
    nb.classify(
        nb.tokenize(
            "hearthstone official professional art. a sorceress, wearing a robe casting a fire ball. insanely coherent physical body parts face, arms, legs, hair, eyes, pupil, eye white. full body realistic, sharp focus, 8k high definition, insanely detailed, intricate, elegant, smooth, sharp focus, illustration, artstation"
        )
    )
)

nb2 = depickle_data("./pickle-db/pickle", "./datasets")
nb2.report_stats()
