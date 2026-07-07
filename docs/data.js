// MSU-Bench Demo Page Data
window.MSU_DATA = {
  videos: [
    {
      id: 'cardhouse',
      name: 'House of Cards',
      type: 'English TV Series',
      file: 'video-example/cardhouse-ep2.mp4',
      annotation: 'annotations/cardhouse-ep2.json',
      thumbnail: 'assets/images/cardhouse-thumb.jpg'
    },
    {
      id: 'mingdynesty',
      name: 'Ming Dynasty 1566',
      type: 'Chinese TV Series',
      file: 'video-example/mingdynesty-ep41.mp4',
      annotation: 'annotations/mingdynesty-ep41.json',
      thumbnail: 'assets/images/mingdynesty-thumb.jpg'
    }
  ],

  tasks: {
    level1: [
      {
        id: 'emotion',
        name: 'Emotion Recognition',
        nameCN: '情感识别',
        description: 'Identify the emotional state of target speakers in multi-speaker conversations',
        icon: '🎭'
      },
      {
        id: 'retrieval',
        name: 'Speaker Retrieval',
        nameCN: '说话人检索',
        description: 'Locate and identify target speakers using various referencing schemes',
        icon: '🔍'
      },
      {
        id: 'counting',
        name: 'Speaker Counting',
        nameCN: '说话人计数',
        description: 'Count the number of distinct speakers in conversational audio',
        icon: '🔢'
      }
    ],
    level2: [
      {
        id: 'emotion_interaction',
        name: 'Emotion Interaction Reasoning',
        nameCN: '多说话人情感交互',
        description: 'Analyze emotional dynamics and influence between speakers',
        icon: '⚡'
      },
      {
        id: 'context',
        name: 'Background Inference',
        nameCN: '对话背景推理',
        description: 'Infer dialogue background and situational context from multi-speaker evidence',
        icon: '🧩'
      },
      {
        id: 'dialogue_act',
        name: 'Dialogue Act Recognition',
        nameCN: '对话行为识别',
        description: 'Identify speech acts and dialogue structure in multi-turn interactions',
        icon: '💬'
      }
    ]
  },

  referencingSchemes: [
    { id: 'no_index', name: 'No Index', description: 'Direct audio snippet as acoustic anchor' },
    { id: 'time_index', name: 'Time Index', description: 'Target speaker specified by time range' },
    { id: 'transcript_index', name: 'Transcript Index', description: 'Target speaker identified via quoted text' },
    { id: 'speaker_index', name: 'Speaker Index', description: 'Target speaker by appearance order' },
    { id: 'complex_index', name: 'Complex Index', description: 'Combined multiple cues for speaker localization' }
  ],

  paperInfo: {
    title: 'MSU-Bench: Towards Speaker-Centric Understanding in Conversational Multi-Speaker Scenarios',
    conference: 'Interspeech 2026',
    authors: 'Zhaokai Sun, Shuai Wang, Zhennan Lin, Chengyou Wang, Dehui Gao, Yu\'ang Cao, Chunjiang He, Pan Zhou, Lei Xie',
    affiliation: 'ASLP@NPU, Northwestern Polytechnical University',
    arxiv: 'https://arxiv.org/abs/2606.22868',
    code: 'https://github.com/ASLP-lab/MSU-Bench',
    demo: 'https://aslp-lab.github.io/msu-bench.github.io/',
    stats: {
      samples: 2300,
      tasks: 16,
      dimensions: 5,
      languages: 2
    }
  }
};

// Load QA data
async function loadQA(videoId, level, taskId) {
  const videoMap = {
    'cardhouse': 'cardhouse-ep2',
    'mingdynesty': 'mingdynesty-ep41'
  };
  const taskMap = {
    'emotion': '说话人属性识别能力-情感识别任务',
    'retrieval': '说话人识别能力-说话人检索任务',
    'counting': '说话人识别能力-说话人计数任务',
    'emotion_interaction': '多说话人上下文推理能力-多说话人情感交互',
    'context': '多说话人对话场景推理能力-对话背景推理',
    'dialogue_act': '多说话人对话结构分析能力-对话行为识别'
  };

  const filePath = `demo_qa/${level}/${videoMap[videoId]}/${taskMap[taskId]}.json`;
  try {
    const response = await fetch(filePath);
    return await response.json();
  } catch (e) {
    console.error('Failed to load QA:', e);
    return null;
  }
}

// Load annotation
async function loadAnnotation(videoId) {
  const videoMap = {
    'cardhouse': 'cardhouse-ep2',
    'mingdynesty': 'mingdynesty-ep41'
  };

  const filePath = `annotations/${videoMap[videoId]}.json`;
  try {
    const response = await fetch(filePath);
    return await response.json();
  } catch (e) {
    console.error('Failed to load annotation:', e);
    return null;
  }
}
