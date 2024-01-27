PREDICTION=/workspace/llama-recipes/training-output/3.0_1/output.txt
TARGET=/workspace/llama-recipes/AMR_function/AMR3.0/test.sequence.target
cd meteor
echo "evaluating METEOR"
java -Xmx2G -jar meteor-*.jar $PREDICTION $TARGET -l en -norm -lower > /workspace/llama-recipes/AMR_evaluate/meteor.txt
cd ../

python eval.py $PREDICTION $TARGET > /workspace/llama-recipes/training-output/3.0_1/evaluation.txt
