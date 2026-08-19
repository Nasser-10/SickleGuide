import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import ForceGraph2D from "react-force-graph-2d";


const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000/api/v1";

const CHAT_STORAGE_KEY =
  "sickleguide_chats_v4";

const THEME_KEY =
  "sickleguide_theme_v3";


const CORE_SOURCES = [
  {
    name: "ASH — Sickle Cell Disease Clinical Practice Guidelines.pdf",
    short:
      "ASH 2020 guidelines for sickle cell disease",
    kind: "Clinical Guideline",
  },
  {
    name: "Evidence-BasedManagement ofSickle Cell Disease.pdf",
    short:
      "Evidence-Based Management of Sickle Cell Disease",
    kind: "Evidence Review",
  },
  {
    name: "Watermarked ASH SCD Transfusion Pocket Guide.pdf",
    short:
      "ASH SCD Transfusion Pocket Guide",
    kind: "Pocket Guide",
  },
  {
    name:
      "WHO consolidated guidelinesfor the management of commonchildhood illness.pdf",
    short:
      "WHO consolidated childhood illness guidelines",
    kind: "WHO Guideline",
  },
  {
    name:
      "WHO recommendations on themanagement of sickle-cell diseaseduring pregnancy, childbirth andthe interpregnancy period.pdf",
    short:
      "WHO SCD pregnancy recommendations",
    kind: "WHO Guideline",
  },
];


function createId() {
  if (
    typeof crypto !== "undefined" &&
    crypto.randomUUID
  ) {
    return crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random()}`;
}


function createChat() {
  return {
    id: createId(),
    title: "New conversation",
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  };
}


function App() {
  // ========================================================
  // Navigation / Theme
  // ========================================================

  const [page, setPage] =
    useState("chat");

  const [theme, setTheme] =
    useState(
      () =>
        localStorage.getItem(
          THEME_KEY
        ) || "light"
    );

  // ========================================================
  // Chat
  // ========================================================

  const [chats, setChats] =
    useState(() => {
      try {
        const saved =
          localStorage.getItem(
            CHAT_STORAGE_KEY
          );

        return saved
          ? JSON.parse(saved)
          : [createChat()];
      } catch {
        return [createChat()];
      }
    });

  const [
    activeChatId,
    setActiveChatId,
  ] = useState(() => {
    try {
      return (
        localStorage.getItem(
          `${CHAT_STORAGE_KEY}_active`
        ) || null
      );
    } catch {
      return null;
    }
  });

  const [query, setQuery] =
    useState("");

  const [loading, setLoading] =
    useState(false);

  const [
    streamingAnswer,
    setStreamingAnswer,
  ] = useState("");

  const [streamStage, setStreamStage] =
    useState("");

  const [
    streamSources,
    setStreamSources,
  ] = useState([]);

  const [error, setError] =
    useState("");

  // ========================================================
  // Graph
  // ========================================================

  const graphRef =
    useRef(null);

  const [graphData, setGraphData] =
    useState({
      nodes: [],
      links: [],
      total_nodes: 0,
      total_edges: 0,
    });

  const [graphView, setGraphView] =
    useState("overview");

  const [
    graphMaxNodes,
    setGraphMaxNodes,
  ] = useState(120);

  const [
    selectedNode,
    setSelectedNode,
  ] = useState(null);

  const [
    visibleEntityTypes,
    setVisibleEntityTypes,
  ] = useState([]);

  const [
    visibleRelations,
    setVisibleRelations,
  ] = useState([]);

  const [
    allGraphEntityTypes,
    setAllGraphEntityTypes,
  ] = useState([]);

  const [
    allGraphRelations,
    setAllGraphRelations,
  ] = useState([]);

  const [
    graphLoading,
    setGraphLoading,
  ] = useState(false);

  const [
    graphError,
    setGraphError,
  ] = useState("");

  // ========================================================
  // Dataset
  // ========================================================

  const [
    dataFiles,
    setDataFiles,
  ] = useState([]);

  const [
    dataStats,
    setDataStats,
  ] = useState({
    total_files: 0,
    total_chunks: 0,
  });

  const [
    uploadLoading,
    setUploadLoading,
  ] = useState(false);

  const [
    uploadMessage,
    setUploadMessage,
  ] = useState("");

  const [
    dragActive,
    setDragActive,
  ] = useState(false);

  // ========================================================
  // Evaluation
  // ========================================================

  const [
    evaluationMode,
    setEvaluationMode,
  ] = useState(false);

  const [
    evaluationLoading,
    setEvaluationLoading,
  ] = useState(false);

  const [
    evaluationResult,
    setEvaluationResult,
  ] = useState(null);

  // ========================================================
  // Persistence
  // ========================================================

  useEffect(() => {
    document.documentElement.dataset.theme =
      theme;

    localStorage.setItem(
      THEME_KEY,
      theme
    );
  }, [theme]);


  useEffect(() => {
    localStorage.setItem(
      CHAT_STORAGE_KEY,
      JSON.stringify(chats)
    );
  }, [chats]);


  useEffect(() => {
    if (activeChatId) {
      localStorage.setItem(
        `${CHAT_STORAGE_KEY}_active`,
        activeChatId
      );
    }
  }, [activeChatId]);


  useEffect(() => {
    if (
      !activeChatId ||
      !chats.some(
        (chat) =>
          chat.id ===
          activeChatId
      )
    ) {
      setActiveChatId(
        chats[0]?.id || null
      );
    }
  }, [
    chats,
    activeChatId,
  ]);


  const activeChat = useMemo(
    () =>
      chats.find(
        (chat) =>
          chat.id === activeChatId
      ) || null,
    [chats, activeChatId]
  );

  // ========================================================
  // Generic helpers
  // ========================================================

  const updateChat = (
    chatId,
    updater
  ) => {
    setChats((current) =>
      current.map((chat) =>
        chat.id === chatId
          ? updater(chat)
          : chat
      )
    );
  };


  // ========================================================
  // Chat helpers
  // ========================================================

  const createNewChat = () => {
    const chat = createChat();

    setChats((current) => [
      chat,
      ...current,
    ]);

    setActiveChatId(chat.id);
    setPage("chat");

    setQuery("");
    setError("");
    setStreamingAnswer("");
    setStreamSources([]);
    setStreamStage("");
  };


  const deleteChat = (
    chatId
  ) => {
    const remaining =
      chats.filter(
        (chat) =>
          chat.id !== chatId
      );

    if (!remaining.length) {
      const fresh = createChat();

      setChats([fresh]);
      setActiveChatId(
        fresh.id
      );

      return;
    }

    setChats(remaining);

    if (
      activeChatId === chatId
    ) {
      setActiveChatId(
        remaining[0].id
      );
    }
  };


  const renameChat = (
    chatId
  ) => {
    const chat =
      chats.find(
        (item) =>
          item.id === chatId
      );

    if (!chat) return;

    const title =
      window.prompt(
        "Conversation name:",
        chat.title
      );

    if (
      !title ||
      !title.trim()
    ) {
      return;
    }

    updateChat(
      chatId,
      (current) => ({
        ...current,
        title:
          title.trim(),
        updatedAt:
          Date.now(),
      })
    );
  };


  // ========================================================
  // Chat streaming
  // ========================================================

  const askSickleGuide =
    async () => {
      const cleanQuery =
        query.trim();

      if (
        !cleanQuery ||
        loading ||
        !activeChat
      ) {
        return;
      }

      setLoading(true);
      setError("");
      setStreamingAnswer("");
      setStreamSources([]);

      setStreamStage(
        "Searching clinical evidence..."
      );

      const userMessage = {
        id: createId(),
        role: "user",
        content: cleanQuery,
        createdAt: Date.now(),
      };

      const history =
        activeChat.messages.map(
          (message) => ({
            role: message.role,
            content:
              message.content,
          })
        );

      updateChat(
        activeChat.id,
        (current) => ({
          ...current,

          title:
            current.messages.length === 0
              ? cleanQuery.length > 45
                ? `${cleanQuery.slice(
                    0,
                    45
                  )}...`
                : cleanQuery
              : current.title,

          messages: [
            ...current.messages,
            userMessage,
          ],

          updatedAt:
            Date.now(),
        })
      );

      setQuery("");

      try {
        const response =
          await fetch(
            `${API_BASE_URL}/chat/stream`,
            {
              method: "POST",

              headers: {
                "Content-Type":
                  "application/json",

                Accept:
                  "text/event-stream",
              },

              body: JSON.stringify({
                query: cleanQuery,
                chat_id:
                  activeChat.id,
                history,
              }),
            }
          );

        if (!response.ok) {
          throw new Error(
            await response.text()
          );
        }

        if (!response.body) {
          throw new Error(
            "Streaming is unavailable."
          );
        }

        const reader =
          response.body.getReader();

        const decoder =
          new TextDecoder();

        let buffer = "";
        let finalAnswer = "";
        let finalSources = [];

        while (true) {
          const {
            value,
            done,
          } =
            await reader.read();

          if (done) break;

          buffer += decoder.decode(
            value,
            {
              stream: true,
            }
          );

          const events =
            buffer.split(
              "\n\n"
            );

          buffer =
            events.pop() || "";

          for (
            const event of events
          ) {
            const line =
              event
                .split("\n")
                .find(
                  (item) =>
                    item.startsWith(
                      "data:"
                    )
                );

            if (!line) continue;

            const payload =
              JSON.parse(
                line
                  .slice(5)
                  .trim()
              );

            if (
              payload.type ===
              "status"
            ) {
              setStreamStage(
                payload.message
              );
            }

            if (
              payload.type ===
              "token"
            ) {
              finalAnswer +=
                payload.content;

              setStreamingAnswer(
                finalAnswer
              );

              setStreamStage(
                "Streaming verified answer..."
              );
            }

            if (
              payload.type ===
              "sources"
            ) {
              finalSources =
                payload.sources ||
                [];

              setStreamSources(
                finalSources
              );
            }

            if (
              payload.type ===
              "error"
            ) {
              throw new Error(
                payload.message ||
                  "Streaming failed."
              );
            }

            if (
              payload.type ===
              "done"
            ) {
              setStreamStage(
                "Complete"
              );
            }
          }
        }

        if (finalAnswer) {
          updateChat(
            activeChat.id,
            (current) => ({
              ...current,

              messages: [
                ...current.messages,

                {
                  id: createId(),
                  role: "assistant",
                  content:
                    finalAnswer,
                  sources:
                    finalSources,
                  createdAt:
                    Date.now(),
                },
              ],

              updatedAt:
                Date.now(),
            })
          );
        }
      } catch (err) {
        setError(
          err?.message ||
            "Could not connect to SickleGuide."
        );
      } finally {
        setLoading(false);
        setStreamStage("");
      }
    };


  const handleKeyDown =
    (event) => {
      if (
        event.key ===
          "Enter" &&
        !event.shiftKey
      ) {
        event.preventDefault();

        askSickleGuide();
      }
    };


  // ========================================================
  // Graph
  // ========================================================

  const loadGraph =
    async () => {
      setGraphLoading(true);
      setGraphError("");

      try {
        const params =
          new URLSearchParams();

        params.set(
          "view",
          graphView
        );

        params.set(
          "max_nodes",
          String(
            graphMaxNodes
          )
        );

        const response =
          await fetch(
            `${API_BASE_URL}/graph?${params}`
          );

        const data =
          await response.json();

        if (!response.ok) {
          throw new Error(
            data.detail ||
              "Graph request failed."
          );
        }

        setGraphData({
          nodes:
            data.nodes || [],

          links:
            data.edges || [],

          total_nodes:
            data.total_nodes || 0,

          total_edges:
            data.total_edges || 0,
        });

        const types =
          data.available_entity_types ||
          [];

        const relations =
          data.available_relations ||
          [];

        setAllGraphEntityTypes(
          types
        );

        setAllGraphRelations(
          relations
        );

        if (
          !visibleEntityTypes.length
        ) {
          setVisibleEntityTypes(
            types
          );
        }

        if (
          !visibleRelations.length
        ) {
          setVisibleRelations(
            relations
          );
        }

        setSelectedNode(null);
      } catch (err) {
        setGraphError(
          err?.message ||
            "Could not load the knowledge graph."
        );
      } finally {
        setGraphLoading(false);
      }
    };


  useEffect(() => {
    if (page === "graph") {
      loadGraph();
    }

    if (page === "data") {
      loadData();
    }
  }, [page]);


  const graphFilteredData =
    useMemo(() => {
      const nodes =
        graphData.nodes.filter(
          (node) =>
            visibleEntityTypes.includes(
              node.type
            )
        );

      const nodeIds =
        new Set(
          nodes.map(
            (node) =>
              node.id
          )
        );

      const links =
        graphData.links.filter(
          (link) =>
            visibleRelations.includes(
              link.relation
            ) &&
            nodeIds.has(
              typeof link.source ===
                "object"
                ? link.source.id
                : link.source
            ) &&
            nodeIds.has(
              typeof link.target ===
                "object"
                ? link.target.id
                : link.target
            )
        );

      return {
        nodes,
        links,
      };
    }, [
      graphData,
      visibleEntityTypes,
      visibleRelations,
    ]);


  const neighborsOfSelected =
    useMemo(() => {
      if (!selectedNode) {
        return [];
      }

      const neighborIds =
        new Set();

      graphFilteredData.links.forEach(
        (link) => {
          const sourceId =
            typeof link.source ===
            "object"
              ? link.source.id
              : link.source;

          const targetId =
            typeof link.target ===
            "object"
              ? link.target.id
              : link.target;

          if (
            sourceId ===
            selectedNode.id
          ) {
            neighborIds.add(
              targetId
            );
          }

          if (
            targetId ===
            selectedNode.id
          ) {
            neighborIds.add(
              sourceId
            );
          }
        }
      );

      return graphFilteredData.nodes.filter(
        (node) =>
          neighborIds.has(
            node.id
          )
      );
    }, [
      selectedNode,
      graphFilteredData,
    ]);


  const toggleEntityType = (
    type
  ) => {
    setVisibleEntityTypes(
      (current) =>
        current.includes(type)
          ? current.filter(
              (item) =>
                item !== type
            )
          : [
              ...current,
              type,
            ]
    );
  };


  const toggleRelation = (
    relation
  ) => {
    setVisibleRelations(
      (current) =>
        current.includes(
          relation
        )
          ? current.filter(
              (item) =>
                item !== relation
            )
          : [
              ...current,
              relation,
            ]
    );
  };


  const selectAllEntityTypes =
    () => {
      setVisibleEntityTypes(
        allGraphEntityTypes
      );
    };


  const clearEntityTypes =
    () => {
      setVisibleEntityTypes([]);
    };


  const selectAllRelations =
    () => {
      setVisibleRelations(
        allGraphRelations
      );
    };


  const clearRelations =
    () => {
      setVisibleRelations([]);
    };


  const fitGraph = () => {
    graphRef.current?.zoomToFit(
      500,
      55
    );
  };


  const focusNode = (
    node
  ) => {
    if (!node) return;

    setSelectedNode(node);

    graphRef.current?.centerAt(
      node.x,
      node.y,
      500
    );

    graphRef.current?.zoom(
      4,
      500
    );
  };


  // ========================================================
  // Dataset
  // ========================================================

  const loadData =
    async () => {
      try {
        const response =
          await fetch(
            `${API_BASE_URL}/data`
          );

        const data =
          await response.json();

        if (!response.ok) {
          throw new Error(
            data.detail ||
              "Dataset loading failed."
          );
        }

        setDataFiles(
          data.files || []
        );

        setDataStats({
          total_files:
            data.total_files ||
            0,

          total_chunks:
            data.total_chunks ||
            0,
        });
      } catch (err) {
        setError(
          err?.message ||
            "Could not load dataset."
        );
      }
    };


  const uploadPDF =
    async (file) => {
      if (!file) return;

      if (
        !file.name
          .toLowerCase()
          .endsWith(".pdf")
      ) {
        setUploadMessage(
          "Only PDF files are supported."
        );

        return;
      }

      setUploadLoading(true);

      setUploadMessage(
        "Adding PDF to the SickleGuide knowledge base..."
      );

      try {
        const formData =
          new FormData();

        formData.append(
          "file",
          file
        );

        const response =
          await fetch(
            `${API_BASE_URL}/data/upload`,
            {
              method: "POST",
              body: formData,
            }
          );

        const data =
          await response.json();

        if (!response.ok) {
          throw new Error(
            data.detail ||
              "Upload failed."
          );
        }

        setUploadMessage(
          `✓ ${data.filename} added successfully — ${data.chunks_added} chunks indexed.`
        );

        await loadData();
      } catch (err) {
        setUploadMessage(
          err?.message ||
            "Upload failed."
        );
      } finally {
        setUploadLoading(false);
      }
    };


  // ========================================================
  // Evaluation
  // ========================================================

  const runEvaluation =
    async () => {
      setEvaluationLoading(
        true
      );

      setEvaluationResult(
        null
      );

      try {
        const response =
          await fetch(
            `${API_BASE_URL}/evaluation/run`,
            {
              method: "POST",

              headers: {
                "Content-Type":
                  "application/json",
              },

              body: JSON.stringify({
                full:
                  evaluationMode,
              }),
            }
          );

        const data =
          await response.json();

        if (!response.ok) {
          throw new Error(
            data.detail ||
              "Evaluation failed."
          );
        }

        setEvaluationResult(
          data.report
        );
      } catch (err) {
        setError(
          err?.message ||
            "Evaluation failed."
        );
      } finally {
        setEvaluationLoading(
          false
        );
      }
    };


  // ========================================================
  // Sidebar
  // ========================================================

  const renderSidebar =
    () => (
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="logo-mark">
            S
          </div>

          <div>
            <div className="sidebar-title">
              SickleGuide
            </div>

            <div className="sidebar-subtitle">
              Clinical Evidence AI
            </div>
          </div>
        </div>

        <button
          className="new-chat-button"
          onClick={
            createNewChat
          }
        >
          + New conversation
        </button>

        <nav className="main-nav">
          {[
            [
              "chat",
              "Chat",
              "◌",
            ],

            [
              "graph",
              "Knowledge Graph",
              "◇",
            ],

            [
              "pipeline",
              "Project Pipeline",
              "↗",
            ],

            [
              "data",
              "Knowledge Base",
              "▣",
            ],

            [
              "evaluation",
              "Evaluation Lab",
              "◎",
            ],
          ].map(
            (item) => (
              <button
                key={
                  item[0]
                }
                className={
                  page ===
                  item[0]
                    ? "nav-item active"
                    : "nav-item"
                }
                onClick={() =>
                  setPage(
                    item[0]
                  )
                }
              >
                <span>
                  {
                    item[2]
                  }
                </span>

                {
                  item[1]
                }
              </button>
            )
          )}
        </nav>

        <div className="sidebar-section-label">
          Conversations
        </div>

        <div className="chat-history">
          {chats
            .slice()
            .sort(
              (a, b) =>
                b.updatedAt -
                a.updatedAt
            )
            .map(
              (chat) => (
                <div
                  key={
                    chat.id
                  }
                  className={
                    chat.id ===
                    activeChatId
                      ? "history-item active"
                      : "history-item"
                  }
                >
                  <button
                    onClick={() => {
                      setActiveChatId(
                        chat.id
                      );

                      setPage(
                        "chat"
                      );
                    }}
                  >
                    {
                      chat.title
                    }
                  </button>

                  <div className="history-actions">
                    <button
                      onClick={() =>
                        renameChat(
                          chat.id
                        )
                      }
                    >
                      ⋯
                    </button>

                    <button
                      onClick={() =>
                        deleteChat(
                          chat.id
                        )
                      }
                    >
                      ×
                    </button>
                  </div>
                </div>
              )
            )}
        </div>

        <div className="sidebar-footer">
          <button
            className="theme-button"
            onClick={() =>
              setTheme(
                theme ===
                  "light"
                  ? "dark"
                  : "light"
              )
            }
          >
            {theme ===
            "light"
              ? "☾ Dark mode"
              : "☀ Light mode"}
          </button>

          <div className="model-info">
            Qwen 2.5 7B • Ollama
          </div>
        </div>
      </aside>
    );


  // ========================================================
  // Chat page
  // ========================================================

  const renderChat =
    () => (
      <>
        <div className="page-toolbar chat-toolbar">
          <div>
            <div className="eyebrow">
              CLINICAL CHAT
            </div>

            <h1>
              {
                activeChat?.title ||
                "SickleGuide"
              }
            </h1>
          </div>

          <div className="status-pill">
            <span className="online-dot" />
            Local AI
          </div>
        </div>

        <div className="chat-area">
          {!activeChat ||
          activeChat.messages.length ===
            0 ? (
            <div className="chat-empty">
              <div className="large-logo">
                S
              </div>

              <div className="gradient-badge">
                Evidence → Retrieval → Verification
              </div>

              <h2>
                Evidence, not guesswork.
              </h2>

              <p>
                SickleGuide answers only from
                retrieved evidence in your clinical
                knowledge base.
              </p>

              <div className="quick-grid">
                {[
                  "What treatments were evaluated for acute chest syndrome in people with sickle cell disease?",

                  "What is recommended for secondary stroke prevention in children and adolescents with sickle cell disease?",

                  "What are the recommendations for fluid management in pregnant women with sickle cell disease?",
                ].map(
                  (example) => (
                    <button
                      key={
                        example
                      }
                      onClick={() =>
                        setQuery(
                          example
                        )
                      }
                    >
                      {
                        example
                      }
                    </button>
                  )
                )}
              </div>
            </div>
          ) : (
            <div className="messages">
              {activeChat.messages.map(
                (message) => (
                  <div
                    key={
                      message.id
                    }
                    className={
                      message.role ===
                      "user"
                        ? "message-row user"
                        : "message-row assistant"
                    }
                  >
                    <div className="message-avatar">
                      {message.role ===
                      "user"
                        ? "You"
                        : "S"}
                    </div>

                    <div className="message-body">
                      <div className="message-role">
                        {message.role ===
                        "user"
                          ? "You"
                          : "SickleGuide"}
                      </div>

                      <div className="message-bubble">
                        {
                          message.content
                        }
                      </div>

                      {message.sources?.length >
                        0 && (
                        <div className="mini-sources">
                          {message.sources.map(
                            (
                              source
                            ) => (
                              <div
                                className="mini-source"
                                key={`${source.citation}-${source.evidence_id}`}
                              >
                                <strong>
                                  [
                                  {
                                    source.evidence_id
                                  }
                                  ]
                                </strong>{" "}
                                {
                                  source.citation
                                }
                              </div>
                            )
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                )
              )}

              {loading && (
                <div className="message-row assistant">
                  <div className="message-avatar">
                    S
                  </div>

                  <div className="message-body">
                    <div className="message-role">
                      SickleGuide
                    </div>

                    <div className="stream-box">
                      <div className="stream-stage">
                        <span className="spinner" />

                        {
                          streamStage
                        }
                      </div>

                      {streamingAnswer && (
                        <div className="stream-answer">
                          {
                            streamingAnswer
                          }

                          <span className="cursor">
                            ▌
                          </span>
                        </div>
                      )}

                      {streamSources.length >
                        0 && (
                        <div className="stream-sources">
                          {streamSources.map(
                            (
                              source
                            ) => (
                              <div
                                key={`${source.citation}-${source.evidence_id}`}
                              >
                                [
                                {
                                  source.evidence_id
                                }
                                ]{" "}
                                {
                                  source.citation
                                }
                              </div>
                            )
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="error-box">
              {
                error
              }
            </div>
          )}
        </div>

        <div className="composer-area">
          <div className="composer">
            <textarea
              value={
                query
              }
              onChange={(
                event
              ) =>
                setQuery(
                  event.target.value
                )
              }
              onKeyDown={
                handleKeyDown
              }
              placeholder="Ask about sickle cell disease..."
              rows={1}
              disabled={
                loading
              }
            />

            <button
              onClick={
                askSickleGuide
              }
              disabled={
                loading ||
                !query.trim()
              }
            >
              {loading
                ? "..."
                : "↑"}
            </button>
          </div>

          <div className="composer-note">
            SickleGuide uses retrieved knowledge
            base evidence only.
          </div>
        </div>
      </>
    );


  // ========================================================
  // Knowledge Graph page
  // ========================================================

  const renderGraph =
    () => (
      <div className="page graph-page">
        <div className="page-toolbar">
          <div>
            <div className="eyebrow">
              KNOWLEDGE GRAPH
            </div>

            <h1>
              Explore SickleGuide
            </h1>

            <p className="page-description">
              Start with a clinically prepared graph.
              Drag, zoom, focus and turn concepts on
              or off without needing to know entity names.
            </p>
          </div>

          <div className="graph-actions">
            <button
              className="secondary-button"
              onClick={
                fitGraph
              }
            >
              Fit
            </button>

            <button
              className="primary-button"
              onClick={
                loadGraph
              }
            >
              Refresh
            </button>
          </div>
        </div>

        <div className="graph-workspace">
          <aside className="graph-sidebar">
            <div className="control-block">
              <label>
                Graph view
              </label>

              <div className="view-options">
                {[
                  [
                    "overview",
                    "Clinical Overview",
                  ],
                  [
                    "treatments",
                    "Treatments",
                  ],
                  [
                    "complications",
                    "Complications",
                  ],
                  [
                    "transfusion",
                    "Transfusion",
                  ],
                  [
                    "pregnancy",
                    "Pregnancy",
                  ],
                ].map(
                  (view) => (
                    <button
                      key={
                        view[0]
                      }
                      className={
                        graphView ===
                        view[0]
                          ? "view-option active"
                          : "view-option"
                      }
                      onClick={() => {
                        setGraphView(
                          view[0]
                        );
                      }}
                    >
                      <span>
                        {graphView ===
                        view[0]
                          ? "●"
                          : "○"}
                      </span>

                      {
                        view[1]
                      }
                    </button>
                  )
                )}
              </div>
            </div>

            <div className="control-block">
              <div className="control-heading">
                <label>
                  Entity types
                </label>

                <div className="tiny-actions">
                  <button
                    onClick={
                      selectAllEntityTypes
                    }
                  >
                    All
                  </button>

                  <button
                    onClick={
                      clearEntityTypes
                    }
                  >
                    None
                  </button>
                </div>
              </div>

              <div className="check-list">
                {allGraphEntityTypes.map(
                  (type) => (
                    <label
                      className="check-row"
                      key={
                        type
                      }
                    >
                      <input
                        type="checkbox"
                        checked={visibleEntityTypes.includes(
                          type
                        )}
                        onChange={() =>
                          toggleEntityType(
                            type
                          )
                        }
                      />

                      <span>
                        {
                          type
                        }
                      </span>
                    </label>
                  )
                )}
              </div>
            </div>

            <div className="control-block">
              <div className="control-heading">
                <label>
                  Relations
                </label>

                <div className="tiny-actions">
                  <button
                    onClick={
                      selectAllRelations
                    }
                  >
                    All
                  </button>

                  <button
                    onClick={
                      clearRelations
                    }
                  >
                    None
                  </button>
                </div>
              </div>

              <div className="check-list relations">
                {allGraphRelations
                  .slice(
                    0,
                    14
                  )
                  .map(
                    (relation) => (
                      <label
                        className="check-row"
                        key={
                          relation
                        }
                      >
                        <input
                          type="checkbox"
                          checked={visibleRelations.includes(
                            relation
                          )}
                          onChange={() =>
                            toggleRelation(
                              relation
                            )
                          }
                        />

                        <span>
                          {
                            relation
                          }
                        </span>
                      </label>
                    )
                  )}
              </div>
            </div>

            <div className="control-block">
              <div className="control-heading">
                <label>
                  Visible nodes
                </label>

                <span className="control-value">
                  {
                    graphMaxNodes
                  }
                </span>
              </div>

              <input
                className="range"
                type="range"
                min="40"
                max="250"
                step="10"
                value={
                  graphMaxNodes
                }
                onChange={(
                  event
                ) =>
                  setGraphMaxNodes(
                    Number(
                      event.target
                        .value
                    )
                  )
                }
              />

              <button
                className="control-apply"
                onClick={
                  loadGraph
                }
              >
                Apply view
              </button>
            </div>

            <div className="graph-tip">
              <strong>
                How to use it
              </strong>

              <p>
                Drag any node. Scroll to zoom.
                Drag empty space to pan.
                Click a node to inspect it.
              </p>
            </div>

            {selectedNode && (
              <div className="selected-card">
                <div className="selected-label">
                  SELECTED ENTITY
                </div>

                <h3>
                  {
                    selectedNode.name
                  }
                </h3>

                <span className="selected-type">
                  {
                    selectedNode.type
                  }
                </span>

                <div className="selected-stats">
                  <strong>
                    {
                      neighborsOfSelected.length
                    }
                  </strong>

                  <span>
                    connected concepts
                  </span>
                </div>

                <button
                  className="focus-button"
                  onClick={() =>
                    focusNode(
                      selectedNode
                    )
                  }
                >
                  Focus on entity
                </button>

                <div className="neighbor-list">
                  {neighborsOfSelected
                    .slice(
                      0,
                      8
                    )
                    .map(
                      (
                        neighbor
                      ) => (
                        <button
                          key={
                            neighbor.id
                          }
                          onClick={() =>
                            focusNode(
                              neighbor
                            )
                          }
                        >
                          {
                            neighbor.name
                          }
                        </button>
                      )
                    )}
                </div>
              </div>
            )}
          </aside>

          <div className="graph-canvas-card">
            {graphLoading ? (
              <div className="graph-loading">
                <span className="spinner large" />

                <strong>
                  Building clinical graph...
                </strong>

                <p>
                  Preparing the selected medical view.
                </p>
              </div>
            ) : graphError ? (
              <div className="graph-loading">
                <strong>
                  Graph unavailable
                </strong>

                <p>
                  {
                    graphError
                  }
                </p>
              </div>
            ) : (
              <ForceGraph2D
                ref={
                  graphRef
                }

                graphData={
                  graphFilteredData
                }

                nodeId="id"

                linkSource="source"

                linkTarget="target"

                nodeLabel={(node) =>
                  `${node.name}\nType: ${node.type}`
                }

                linkLabel={(link) =>
                  link.relation ||
                  ""
                }

                nodeColor={(node) => {
                  if (
                    selectedNode?.id ===
                    node.id
                  ) {
                    return "#7c87f2";
                  }

                  const colors = {
                    disease:
                      "#d65365",

                    drug:
                      "#4c8fe8",

                    treatment:
                      "#8861d8",

                    procedure:
                      "#df9751",

                    condition:
                      "#51ad89",

                    symptom:
                      "#d766b2",

                    outcome:
                      "#5baec2",

                    therapy:
                      "#776bd1",

                    "lab test":
                      "#3fa0a0",
                  };

                  return (
                    colors[
                      node.type
                    ] ||
                    "#75829b"
                  );
                }}

                nodeRelSize={
                  6
                }

                linkColor={() =>
                  theme ===
                  "dark"
                    ? "rgba(185,195,220,0.30)"
                    : "rgba(91,105,137,0.28)"
                }

                linkWidth={(
                  link
                ) => {
                  if (
                    !selectedNode
                  ) {
                    return 1;
                  }

                  const sourceId =
                    typeof link.source ===
                    "object"
                      ? link.source.id
                      : link.source;

                  const targetId =
                    typeof link.target ===
                    "object"
                      ? link.target.id
                      : link.target;

                  return (
                    sourceId ===
                      selectedNode.id ||
                    targetId ===
                      selectedNode.id
                  )
                    ? 3
                    : 0.8;
                }}

                linkDirectionalArrowLength={
                  4
                }

                linkDirectionalArrowRelPos={
                  1
                }

                cooldownTicks={
                  90
                }

                d3AlphaDecay={
                  0.03
                }

                d3VelocityDecay={
                  0.24
                }

                onNodeClick={(
                  node
                ) => {
                  setSelectedNode(
                    node
                  );

                  graphRef.current?.centerAt(
                    node.x,
                    node.y,
                    400
                  );
                }}

                nodeCanvasObjectMode={() =>
                  "after"
                }

                nodeCanvasObject={(
                  node,
                  ctx,
                  globalScale
                ) => {
                  const label =
                    node.name;

                  const fontSize =
                    Math.max(
                      9 /
                        globalScale,
                      2.5
                    );

                  ctx.font = `${fontSize}px Inter, sans-serif`;

                  ctx.textAlign =
                    "center";

                  ctx.textBaseline =
                    "middle";

                  ctx.fillStyle =
                    theme ===
                    "dark"
                      ? "#edf2fa"
                      : "#2b3448";

                  ctx.fillText(
                    label.length >
                      24
                      ? `${label.slice(
                          0,
                          24
                        )}…`
                      : label,

                    node.x,

                    node.y + 12
                  );
                }}
              />
            )}

            <div className="graph-canvas-hud">
              <div>
                <strong>
                  {
                    graphFilteredData
                      .nodes.length
                  }
                </strong>

                <span>
                  visible nodes
                </span>
              </div>

              <div>
                <strong>
                  {
                    graphFilteredData
                      .links.length
                  }
                </strong>

                <span>
                  relations
                </span>
              </div>

              <div>
                <strong>
                  {
                    graphView ===
                    "overview"
                      ? "Overview"
                      : graphView
                  }
                </strong>

                <span>
                  current view
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );


  // ========================================================
  // Knowledge Base page
  // ========================================================

  const renderData =
    () => {
      const coreNames =
        new Set(
          CORE_SOURCES.map(
            (item) =>
              item.name
          )
        );

      const uploadedFiles =
        dataFiles.filter(
          (file) =>
            !coreNames.has(
              file.name
            )
        );

      return (
        <div className="page">
          <div className="page-toolbar">
            <div>
              <div className="eyebrow">
                KNOWLEDGE BASE
              </div>

              <h1>
                Clinical evidence
              </h1>

              <p className="page-description">
                These five core documents are the original
                knowledge base SickleGuide was built on.
                Additional PDFs can be added below.
              </p>
            </div>
          </div>

          <div className="dataset-summary">
            <div className="summary-card">
              <span>
                Core documents
              </span>

              <strong>
                {
                  CORE_SOURCES.length
                }
              </strong>
            </div>

            <div className="summary-card">
              <span>
                Total indexed documents
              </span>

              <strong>
                {
                  dataStats.total_files
                }
              </strong>
            </div>

            <div className="summary-card">
              <span>
                Total chunks
              </span>

              <strong>
                {
                  dataStats.total_chunks
                }
              </strong>
            </div>
          </div>

          <section className="source-section">
            <div className="section-heading">
              <div>
                <div className="eyebrow">
                  ORIGINAL DATASET
                </div>

                <h2>
                  SickleGuide Core Sources
                </h2>
              </div>

              <span className="core-badge">
                CORE DATA
              </span>
            </div>

            <div className="dataset-grid">
              {
                CORE_SOURCES.map(
                  (
                    source
                  ) => {
                    const live =
                      dataFiles.find(
                        (
                          file
                        ) =>
                          file.name ===
                          source.name
                      );

                    return (
                      <div
                        className="dataset-card core"
                        key={
                          source.name
                        }
                      >
                        <div className="pdf-icon">
                          PDF
                        </div>

                        <div className="dataset-main">
                          <div className="dataset-badges">
                            <span className="badge-core">
                              CORE SOURCE
                            </span>

                            <span className="badge-type">
                              {
                                source.kind
                              }
                            </span>
                          </div>

                          <h3>
                            {
                              source.short
                            }
                          </h3>

                          <p>
                            {
                              source.name
                            }
                          </p>

                          <div className="dataset-meta">
                            <span>
                              {
                                live?.chunks ??
                                "Indexed"
                              }{" "}
                              chunks
                            </span>

                            <span>
                              Active
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  }
                )
              }
            </div>
          </section>

          <section className="source-section">
            <div className="section-heading">
              <div>
                <div className="eyebrow">
                  EXTEND THE KNOWLEDGE BASE
                </div>

                <h2>
                  Add another clinical PDF
                </h2>
              </div>
            </div>

            <div
              className={
                dragActive
                  ? "dropzone drag-active"
                  : "dropzone"
              }

              onClick={() =>
                document
                  .getElementById(
                    "pdf-upload"
                  )
                  ?.click()
              }

              onDragOver={(
                event
              ) => {
                event.preventDefault();
                setDragActive(
                  true
                );
              }}

              onDragLeave={() =>
                setDragActive(
                  false
                )
              }

              onDrop={(
                event
              ) => {
                event.preventDefault();

                setDragActive(
                  false
                );

                uploadPDF(
                  event
                    .dataTransfer
                    .files?.[0]
                );
              }}
            >
              <div className="dropzone-icon">
                ↑
              </div>

              <div className="dropzone-content">
                <strong>
                  Drop a PDF here
                </strong>

                <p>
                  Or click anywhere in this box to
                  choose a clinical document.
                </p>

                <span>
                  PDF → Parse → Clean → Chunk → Embed → Index
                </span>
              </div>

              <input
                id="pdf-upload"
                type="file"
                accept=".pdf,application/pdf"
                hidden
                disabled={
                  uploadLoading
                }
                onChange={(
                  event
                ) => {
                  uploadPDF(
                    event
                      .target
                      .files?.[0]
                  );

                  event.target.value =
                    "";
                }}
              />

              {uploadLoading && (
                <div className="upload-spinner">
                  Processing...
                </div>
              )}
            </div>

            {uploadMessage && (
              <div className="upload-message">
                {
                  uploadMessage
                }
              </div>
            )}
          </section>

          <section className="source-section">
            <div className="section-heading">
              <div>
                <div className="eyebrow">
                  ADDITIONAL KNOWLEDGE
                </div>

                <h2>
                  Uploaded documents
                </h2>
              </div>

              <span className="uploaded-badge">
                {
                  uploadedFiles.length
                }{" "}
                added
              </span>
            </div>

            {uploadedFiles.length ===
            0 ? (
              <div className="empty-source-card">
                <div className="empty-source-icon">
                  +
                </div>

                <div>
                  <strong>
                    No additional documents yet
                  </strong>

                  <p>
                    The original SickleGuide dataset is
                    already active. Add a PDF above to extend
                    the searchable knowledge base.
                  </p>
                </div>
              </div>
            ) : (
              <div className="dataset-grid">
                {uploadedFiles.map(
                  (
                    file
                  ) => (
                    <div
                      className="dataset-card uploaded"
                      key={
                        file.name
                      }
                    >
                      <div className="pdf-icon uploaded-icon">
                        PDF
                      </div>

                      <div className="dataset-main">
                        <div className="dataset-badges">
                          <span className="badge-uploaded">
                            UPLOADED
                          </span>
                        </div>

                        <h3>
                          {
                            file.name
                          }
                        </h3>

                        <p>
                          {
                            file.size_mb
                          }{" "}
                          MB
                        </p>

                        <div className="dataset-meta">
                          <span>
                            {
                              file.chunks
                            }{" "}
                            chunks
                          </span>

                          <span>
                            Searchable
                          </span>
                        </div>
                      </div>
                    </div>
                  )
                )}
              </div>
            )}
          </section>
        </div>
      );
    };


  // ========================================================
  // Pipeline
  // ========================================================

  const renderPipeline =
    () => {
      const stages = [
        [
          "01",
          "PDF Sources",
          "Curated clinical guidelines and evidence.",
        ],
        [
          "02",
          "Ingestion",
          "Page-aware PDF parsing and metadata extraction.",
        ],
        [
          "03",
          "Cleaning",
          "Remove extraction noise before retrieval.",
        ],
        [
          "04",
          "Markdown",
          "Normalize content into structured Markdown.",
        ],
        [
          "05",
          "Chunking",
          "Context-preserving chunks with citations.",
        ],
        [
          "06",
          "Embeddings",
          "BGE-M3 representations in the vector database.",
        ],
        [
          "07",
          "Hybrid Retrieval",
          "Dense + BM25 + Graph fused with RRF.",
        ],
        [
          "08",
          "Reranking",
          "BGE reranker selects strongest evidence.",
        ],
        [
          "09",
          "Generation",
          "Qwen 2.5 7B generates from evidence only.",
        ],
        [
          "10",
          "Grounding",
          "Unsupported medical claims are rejected.",
        ],
        [
          "11",
          "Citations",
          "Evidence references are validated.",
        ],
        [
          "12",
          "Safety",
          "Fail closed when evidence is insufficient.",
        ],
      ];

      return (
        <div className="page">
          <div className="page-toolbar">
            <div>
              <div className="eyebrow">
                SYSTEM ARCHITECTURE
              </div>

              <h1>
                How SickleGuide works
              </h1>
            </div>
          </div>

          <div className="pipeline-hero">
            <div className="pipeline-orbit">
              RAG
            </div>

            <div>
              <h2>
                From clinical PDF to verified answer.
              </h2>

              <p>
                Every stage improves retrieval quality,
                traceability, grounding and safety.
              </p>
            </div>
          </div>

          <div className="pipeline-grid">
            {
              stages.map(
                (
                  stage
                ) => (
                  <div
                    className="pipeline-card"
                    key={
                      stage[0]
                    }
                  >
                    <div className="stage-number">
                      {
                        stage[0]
                      }
                    </div>

                    <div className="stage-content">
                      <h3>
                        {
                          stage[1]
                        }
                      </h3>

                      <p>
                        {
                          stage[2]
                        }
                      </p>

                      <div className="stage-visual">
                        <span />
                        <span />
                        <span />
                      </div>
                    </div>
                  </div>
                )
              )
            }
          </div>
        </div>
      );
    };


  // ========================================================
  // Evaluation
  // ========================================================

  const renderEvaluation =
    () => {
      const retrieval =
        evaluationResult
          ?.retrieval
          ?.summary;

      const e2e =
        evaluationResult
          ?.end_to_end
          ?.summary;

      return (
        <div className="page">
          <div className="page-toolbar">
            <div>
              <div className="eyebrow">
                QUALITY LAB
              </div>

              <h1>
                Evaluation
              </h1>

              <p className="page-description">
                Measure retrieval, reranking, grounding,
                citations and end-to-end answer quality.
              </p>
            </div>

            <button
              className="primary-button"
              onClick={
                runEvaluation
              }
              disabled={
                evaluationLoading
              }
            >
              {evaluationLoading
                ? "Running..."
                : "Run evaluation"}
            </button>
          </div>

          <div className="evaluation-controls">
            <div>
              <h3>
                Evaluation mode
              </h3>

              <p>
                Retrieval is faster. Full mode also runs
                generation and grounding.
              </p>
            </div>

            <label className="toggle">
              <input
                type="checkbox"
                checked={
                  evaluationMode
                }
                onChange={(
                  event
                ) =>
                  setEvaluationMode(
                    event.target
                      .checked
                  )
                }
              />

              <span />

              Full End-to-End
            </label>
          </div>

          <div className="metric-grid">
            {[
              [
                "Recall@5",
                retrieval
                  ? `${(
                      retrieval[
                        "candidate_recall@5"
                      ] * 100
                    ).toFixed(1)}%`
                  : "—",
              ],

              [
                "Reranked Recall@5",
                retrieval
                  ? `${(
                      retrieval[
                        "reranked_recall@5"
                      ] * 100
                    ).toFixed(1)}%`
                  : "—",
              ],

              [
                "MRR",
                retrieval
                  ? retrieval.mrr.toFixed(
                      3
                    )
                  : "—",
              ],

              [
                "Grounding",
                e2e
                  ? `${(
                      e2e.grounded_rate *
                      100
                    ).toFixed(1)}%`
                  : "—",
              ],

              [
                "Citation validity",
                e2e
                  ? `${(
                      e2e.citation_valid_rate *
                      100
                    ).toFixed(1)}%`
                  : "—",
              ],

              [
                "Answer coverage",
                e2e
                  ? `${(
                      e2e.answer_term_coverage *
                      100
                    ).toFixed(1)}%`
                  : "—",
              ],
            ].map(
              (metric) => (
                <div
                  className="metric-card"
                  key={
                    metric[0]
                  }
                >
                  <span>
                    {
                      metric[0]
                    }
                  </span>

                  <strong>
                    {
                      metric[1]
                    }
                  </strong>
                </div>
              )
            )}
          </div>

          <div className="evaluation-method-grid">
            {[
              [
                "Retrieval",
                "Recall@K, source recall and MRR.",
              ],
              [
                "Reranking",
                "Checks whether useful evidence reaches the top.",
              ],
              [
                "Grounding",
                "Checks medical claims against retrieved evidence.",
              ],
              [
                "Citations",
                "Checks evidence references before output.",
              ],
              [
                "Safety",
                "Prevents unsupported or unsafe behavior.",
              ],
              [
                "End-to-End",
                "Measures the entire SickleGuide pipeline.",
              ],
            ].map(
              (item) => (
                <div
                  className="evaluation-method"
                  key={
                    item[0]
                  }
                >
                  <div className="method-icon">
                    ✓
                  </div>

                  <div>
                    <h3>
                      {
                        item[0]
                      }
                    </h3>

                    <p>
                      {
                        item[1]
                      }
                    </p>
                  </div>
                </div>
              )
            )}
          </div>
        </div>
      );
    };


  // ========================================================
  // Router
  // ========================================================

  const renderPage =
    () => {
      if (
        page === "chat"
      ) {
        return renderChat();
      }

      if (
        page === "graph"
      ) {
        return renderGraph();
      }

      if (
        page === "pipeline"
      ) {
        return renderPipeline();
      }

      if (
        page === "data"
      ) {
        return renderData();
      }

      return renderEvaluation();
    };


  return (
    <div className="app-shell">
      {renderSidebar()}

      <main className="main-shell">
        {renderPage()}
      </main>
    </div>
  );
}


export default App;