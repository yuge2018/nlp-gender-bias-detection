P4的功能与失败输入测试放这里。不要修改其他成员测试。

运行：

```sh
python -m unittest discover -s tests/p4 -v
```

覆盖：metrics 与冻结契约一致、零分母、缺行/重复/乱序/NaN/越界/bool 标签、
阈值不符、mock 标记与数据 hash 不一致、mock 拒绝与真实/mock 混合拒绝、
单次标准差为 null、CLI 退出码 0/2。
