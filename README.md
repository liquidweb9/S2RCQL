# Code For Paper Can LLM be a Good Path Planner based on Prompt Engineering? Mitigating the Hallucination for Path Planning



 ## Comparison of the performance of S2RCQL and Rememberer

<img src="D:\environment\jsbsim_env\gpt-driver\S2RCQL\resource\Fig5.png" alt="Fig5" style="zoom:33%;" />



## Comparison of the effectiveness of the S2RCQL algorithm without course or S2R and under different curriculum generation schemes.

<img src="D:\environment\jsbsim_env\gpt-driver\S2RCQL\resource\Fig6.png" alt="Fig6" style="zoom: 33%;" />

## The results for each model are in all mazes. The best results are highlighted in Bold, and the best baseline models are underlined. The score in each cell is represented as "ChatGLM-6B[ChatGPT score, ERNIE-Bot score]".

| Method     | Success (%) (5×5)     | Optimality (%) (5×5)  | Success (%) (7×7)     | Optimality (%) (7×7)  | Success (%) (10×10)   | Optimality (%) (10×10) | Success (%) (100×100) | Optimality (%) (100×100) |
| ---------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- | ---------------------- | --------------------- | ------------------------ |
| Naive      | 9.2 [11.0, 11.4]      | 9.5 [11.5, 10.8]      | 8.5 [10.6, 10.3]      | 9.3 [13.0, 12.5]      | 7.4 [9.0, 9.1]        | 8.5 [9.2, 8.9]         | -                     | -                        |
| CoT        | 12.5 [15.0, 15.0]     | 13.5 [15.4, 14.5]     | 11.1 [14.2, 14.9]     | 12.8 [14.2, 13.5]     | 8.9 [10.3, 10.5]      | 9.7 [11.7, 10.1]       | -                     | -                        |
| ToT        | 13.7 [16.8, 17.1]     | 14.3 [14.6, 13.8]     | 12.0 [16.5, 16.6]     | 13.7 [14.4, 13.1]     | 9.0 [10.0, 10.3]      | 11.1 [12.5, 12.9]      | -                     | -                        |
| ReAct      | 15.0 [17.6, 17.4]     | 15.8 [23.5, 22.8]     | 14.3 [17.4, 16.1]     | 15.0 [21.6, 21.6]     | 12.6 [14.8, 15.4]     | 14.4 [21.3, 20.7]      | -                     | -                        |
| Remember   | __37.6 [44.7, 45.1]__ | __46.4 [52.6, 50.8]__ | __34.8 [37.4, 44.2]__ | __37.8 [46.4, 40.2]__ | __32.5 [32.6, 34.8]__ | __30.5 [36.6, 35.7]__  | 20.0 [26.7, 26.7]     | 50.0 [50.0, 50.0]        |
| **S2RCQL** | **74.5 [83.5, 85.6]** | **63.8 [74.6, 73.8]** | **63.8 [72.5, 73.4]** | **60.6 [71.6, 69.6]** | **55.6 [63.8, 64.7]** | **56.7 [67.4, 65.7]**  | **50.0 [53.3, 56.7]** | **50.0 [62.5, 58.9]**    |

## Experimental results on robotic navigation tasks.The best results are highlighted by **Bold**, and the best baseline models are marked by _Underline_. The score in each cell is represented as "Success Rate[Optimality Rate]".  

| Method     | ChatGPT 3.5     | ERNIE-Bot 4.0   | ChatGLM-6B      |
| ---------- | --------------- | --------------- | --------------- |
| Naive      | 11.0 [4.5]      | 11.4 [4.0]      | 9.2 [3.5]       |
| CoT        | 15.0 [5.7]      | 15.0 [5.5]      | 12.5 [4.5]      |
| ToT        | 16.4 [5.6]      | 17.0 [4.5]      | 12.3 [4.5]      |
| ReAct      | 17.7 [10.5]     | 16.5 [9.8]      | 15.0 [7.8]      |
| _Rem_      | __46.4 [32.6]__ | __42.1 [30.3]__ | __37.9 [28.6]__ |
| **S2RCQL** | **72.5 [66.7]** | **74.6 [63.8]** | **65.3 [56.7]** |
