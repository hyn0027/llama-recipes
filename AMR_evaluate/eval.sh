PREDICTION=/home/hyn0027/workspace/llama-recipes/training-output/silver_3.0_0/output.txt
TARGET=/home/hyn0027/workspace/llama-recipes/AMR_function/AMR3.0/test.sequence.target
cd meteor
echo "evaluating METEOR"
java -Xmx2G -jar meteor-*.jar $PREDICTION $TARGET -l en -norm -lower > /home/hyn0027/workspace/llama-recipes/AMR_evaluate/meteor.txt
cd ../

python eval.py $PREDICTION $TARGET > /home/hyn0027/workspace/llama-recipes/training-output/silver_3.0_0/evaluation.txt
