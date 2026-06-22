"""
maxinfo/data.py

jaco_play(Open-X, RLDS)를 **로컬 샤드**에서 읽어 (image, instruction, 7D normalized action)
서브셋을 메모리에 적재한다. (GCS 직접 스트리밍은 ~1.8MB/s로 느려, 샤드를 로컬에 받아 사용)

OpenVLA의 jaco_play 변환/정규화를 정확히 재현:
  action_7d = [world_vector(3), zeros(3), gripper_abs(1)]
  정규화    = predict_action 역연산(q01/q99, mask) 정합
"""
import os, json, time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

GCS_DIR = "gs://gresearch/robotics/jaco_play/0.1.0"
LOCAL_DIR = "data/jaco_play/0.1.0"
DS_KEY = "jaco_play"


def ensure_shards(n_train_shards=4, n_test_shards=1):
    """필요한 메타 + 샤드를 로컬에 준비(없으면 GCS에서 다운로드)."""
    os.makedirs(LOCAL_DIR, exist_ok=True)
    for j in ["dataset_info.json", "features.json"]:
        if not os.path.exists(f"{LOCAL_DIR}/{j}"):
            tf.io.gfile.copy(f"{GCS_DIR}/{j}", f"{LOCAL_DIR}/{j}", overwrite=True)
    train = sorted(tf.io.gfile.glob(GCS_DIR + "/jaco_play-train.tfrecord*"))
    test = sorted(tf.io.gfile.glob(GCS_DIR + "/jaco_play-test.tfrecord*"))
    want = train[:n_train_shards] + test[:n_test_shards]
    local = []
    for s in want:
        dst = f"{LOCAL_DIR}/{s.split('/')[-1]}"
        if not os.path.exists(dst):
            t = time.time(); tf.io.gfile.copy(s, dst, overwrite=True)
            print(f"  downloaded {os.path.basename(dst)} "
                  f"({os.path.getsize(dst)/1e6:.0f}MB, {time.time()-t:.0f}s)", flush=True)
        local.append(dst)
    train_local = [f for f in local if "-train." in f]
    test_local = [f for f in local if "-test." in f]
    return train_local, test_local


def _rel2abs_gripper(actions: tf.Tensor) -> tf.Tensor:
    """OpenVLA rel2abs_gripper_actions 정확 재현 (1D, trajectory 단위)."""
    opening_mask, closing_mask = actions < -0.1, actions > 0.1
    th = tf.where(opening_mask, 1, tf.where(closing_mask, -1, 0))
    start = -1 * th[tf.argmax(th != 0, axis=0)]
    start = tf.cond(start == 0, lambda: 1, lambda: start)

    def scan_fn(carry, i):
        return tf.cond(th[i] == 0, lambda: carry, lambda: th[i])

    new = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)
    return tf.cast(new, tf.float32) / 2 + 0.5


def _normalizer(config_path):
    a = json.load(open(config_path))["norm_stats"][DS_KEY]["action"]
    q01 = np.array(a["q01"], np.float32); q99 = np.array(a["q99"], np.float32)
    mask = np.array(a["mask"], bool); rng = q99 - q01

    def norm(act7):
        out = act7.copy().astype(np.float32)
        safe = rng > 1e-8; idx = mask & safe
        out[idx] = np.clip(2 * (act7[idx] - q01[idx]) / rng[idx] - 1, -1, 1)
        out[mask & ~safe] = 0.0
        return out
    return norm


def _episodes(shards):
    """로컬 tfrecord 샤드 -> 에피소드(steps 중첩 Dataset 포함) eager 반복."""
    builder = tfds.builder_from_directory(LOCAL_DIR)
    feats = builder.info.features
    ds = tf.data.TFRecordDataset(shards).map(feats.deserialize_example)
    for ep in ds:
        yield list(ep["steps"])  # step dict(tensor) 리스트


def load_jaco_subset(n_train=1500, n_val=256, config_path="my_openvla_honeybee/config.json",
                     seed=0, n_train_shards=4, n_test_shards=1):
    norm = _normalizer(config_path)
    train_shards, test_shards = ensure_shards(n_train_shards, n_test_shards)

    def collect(shards, need):
        imgs, ins, acts = [], [], []
        for steps in _episodes(shards):
            wv = np.stack([s["action"]["world_vector"].numpy() for s in steps]).astype(np.float32)
            grip = np.stack([float(s["action"]["gripper_closedness_action"].numpy()[0]) for s in steps]).astype(np.float32)
            grip_abs = _rel2abs_gripper(tf.constant(grip)).numpy()
            act7 = np.concatenate([wv, np.zeros_like(wv), grip_abs[:, None]], -1)
            for t, s in enumerate(steps):
                imgs.append(s["observation"]["image"].numpy())
                instr = s["observation"]["natural_language_instruction"].numpy()
                ins.append(instr.decode() if isinstance(instr, bytes) else str(instr))
                acts.append(norm(act7[t]))
            if len(imgs) >= need:
                break
        return imgs, ins, acts

    tr_i, tr_s, tr_a = collect(train_shards, n_train)
    va_i, va_s, va_a = collect(test_shards, n_val)
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(tr_i))[:n_train]
    train = [(tr_i[i], tr_s[i], tr_a[i]) for i in p]
    val = list(zip(va_i[:n_val], va_s[:n_val], va_a[:n_val]))
    return train, val


if __name__ == "__main__":
    tr, va = load_jaco_subset(n_train=30, n_val=10)
    print(f"train={len(tr)} val={len(va)}")
    img, ins, act = tr[0]
    print("image", img.shape, img.dtype, "instr:", repr(ins))
    print("norm action:", [round(float(x), 4) for x in act])
    arr = np.stack([a for _, _, a in tr])
    print("action mean", arr.mean(0).round(3), "min", arr.min(0).round(2), "max", arr.max(0).round(2))
