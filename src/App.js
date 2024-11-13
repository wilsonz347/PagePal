// src/App.js

function Header() {
  return (
    <header className="bg-blue-600 text-white p-4 text-center">
      <h1 className="text-3xl">My React Application</h1>
    </header>
  );
}

function MainContent() {
  return (
    <main className="p-8">
      <h2 className="text-2xl font-semibold mb-4">Welcome to the App</h2>
      <p className="text-lg">This is a simple layout example using Tailwind CSS and React.</p>
    </main>
  );
}

function Footer() {
  return (
    <footer className="bg-gray-800 text-white p-4 text-center mt-8">
      <p>&copy; 2024 My React App</p>
    </footer>
  );
}

function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <MainContent />
      <Footer />
    </div>
  );
}

export default App;