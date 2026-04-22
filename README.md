# BA-SGCL
Official code for the paper **["Adversarial Robustness of Link Sign Prediction in Signed Graphs"](https://arxiv.org/abs/2401.10590)**.

In the folder, we provide the clean signed graph datasets mentioned in our paper. To evaluate BA-SGCL's performance, it is recommended to apply Balance-attack and FlipAttack on the clean datasets beforehand. Change the input data both in `code/train.py` and `code/utils/edge_data_sign.py`.

# Run
```bash
cd code
python train.py
```

# Citation
If you find this repository useful, please cite:

```bibtex
@article{zhou2024adversarial,
  title={Adversarial Robustness of Link Sign Prediction in Signed Graphs},
  author={Zhou, Jialong and Ai, Xing and Lai, Yuni and Michalak, Tomasz and Li, Gaolei and Li, Jianhua and Tang, Di and Zhang, Xingxing and Yang, Mengpei and Zhou, Kai},
  journal={arXiv preprint arXiv:2401.10590},
  year={2024}
}
```
