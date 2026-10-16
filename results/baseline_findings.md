# Baseline Findings

## Classification (macro-averaged, 5-fold grouped cross-validation, 724 comments)

              model  precision  recall  f1-score  accuracy  f1_fold_mean  f1_fold_std
           category      0.639   0.624     0.626     0.624         0.628        0.017
category_with_topic      0.759   0.750     0.752     0.751         0.751        0.014
          sentiment      0.737   0.621     0.656     0.768         0.651        0.085
 sentiment_balanced      0.722   0.750     0.734     0.794         0.737        0.036

## TF-IDF + KMeans (k=6)

 cluster  size                                                                                   top_terms
       0    90             good, students, know, facility, faculties, faculty, fair, faults, feature, felt
       1    68  university, activities, best, extracurricular, good, students, terms, held, lot, regarding
       2   484                  good, books, lab, library, course, students, content, knowledge, work, bad
       3    41 pattern, checking, paper, exam, marks, good, examination, distribution, improved, questions
       4    15              average, way, courses, term, based, content, teaching, subjects, faculty, fair
       5    26     excellent, facilities, library, going, lectures, usefull, work, depends, simply, superb

